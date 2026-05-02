from collections import OrderedDict

import torch

from .constants import Constants
from .defaults import Defaults
from .variable import Variable
from .observed import Observed
from .deterministic import Deterministic
from .selection import Selection
from .proposal import Proposal
from .step import Step
from .plate import Plate
from .blanket import Blanket
from .factorgraph import Factor
from .factorgraph import FactorGraph

class Model():

    class _TraceTensor(torch.Tensor):
        """
        Wrapper around torch.Tensor that tracks the variables in the hierarchical model
        to build the computational graph and dependencies during the build phase.
        """

        @staticmethod
        def __new__(cls, value, *, site=None, parents=None):
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            obj = torch.Tensor._make_subclass(cls, tensor, require_grad=tensor.requires_grad)
            object.__setattr__(obj, "_trace_site", site)
            if parents is None:
                parents = [] if site is None else [site]
            object.__setattr__(obj, "_trace_parents", cls._deduplicate_sites(parents))
            return obj

        @staticmethod
        def _deduplicate_sites(sites):
            """
            Remove duplicate sites from the list while preserving order.
            """

            unique_sites = []
            seen = set()
            for current_site in sites:
                key = id(current_site)
                if key in seen:
                    continue
                seen.add(key)
                unique_sites.append(current_site)
            return unique_sites

        @property
        def parents(self):
            """
            Return a list of parent sites for this tensor.
            """

            return list(self._trace_parents)

        def raw(self):
            """
            Return the raw tensor value without the trace wrapper.
            """

            return self.as_subclass(torch.Tensor)

        @classmethod
        def _unwrap(cls, value, parents):
            """
            Recursively unwrap the value, extracting the raw tensors and collecting parent sites.
            """

            if isinstance(value, cls):
                parents.extend(value.parents)
                return value.raw()
            if isinstance(value, list):
                return [cls._unwrap(item, parents) for item in value]
            if isinstance(value, tuple):
                return tuple(cls._unwrap(item, parents) for item in value)
            if isinstance(value, dict):
                return {key: cls._unwrap(item, parents) for key, item in value.items()}
            return value

        @classmethod
        def _wrap(cls, value, parents):
            """
            Recursively wrap the value, creating trace tensors and associating parent sites.
            """

            if isinstance(value, torch.Tensor):
                return cls(value, parents=parents)
            if isinstance(value, list):
                return [cls._wrap(item, parents) for item in value]
            if isinstance(value, tuple):
                return tuple(cls._wrap(item, parents) for item in value)
            if isinstance(value, dict):
                return {key: cls._wrap(item, parents) for key, item in value.items()}
            return value

        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            """
            Intercept torch function calls to unwrap trace tensors, execute the function
            on raw tensors, and wrap the result back into trace tensors.
            """

            if kwargs is None:
                kwargs = {}

            parents = []
            raw_args = cls._unwrap(args, parents)
            raw_kwargs = cls._unwrap(kwargs, parents)
            result = func(*raw_args, **raw_kwargs)
            return cls._wrap(result, cls._deduplicate_sites(parents))

    class _Context():
        """
        Base context class for managing the state and operations within a model.
        """

        def __init__(self, model, state={}):
            self.__model = model
            self.__state = state
            self._plate_stack = []

        #region Properties

        def __get_state(self):
            return self.__state
        
        state = property(__get_state)

        def __get_model(self):
            return self.__model
        
        model = property(__get_model)

        #endregion

        def sample(self, name, dist, observed=False):
            raise NotImplementedError()
        
        def plate(self, name, size):
            """
            Create or retrieve a plate with the given name and size. The models calls it
            when entering a 'with model.plate(...)' block in the model definition.
            """

            if name not in self.model.plates:
                plate = Plate(name, size, stack=self._plate_stack)
                self.model.plates[name] = plate
            else:
                plate = self.model.plates[name]
                plate.set_stack(self._plate_stack)

            return plate

        def select(self, name, values, indices):
            raise NotImplementedError()

        def step(self, name, sites, /, proposal=Constants.MISSING, log_prob_func=Constants.MISSING, propose_func=Constants.MISSING, update_func=Constants.MISSING):
            # This is a no-op, only used in the build context
            pass

        def _distribution_batch_shape(self, parent_plates, batch_shape=None):
            """
            Compute the batch shape for a distribution based on the currently
            active plates in the context. Exclude the plates that are shared with
            the parent sites since those will already return expanded tensors.
            """

            if batch_shape is None:
                batch_shape = []
            else:
                batch_shape = list(batch_shape)

            for plate in self._plate_stack:
                if plate not in parent_plates:
                    for size in plate.size:
                        batch_shape.append(size)

            return tuple(batch_shape)
            
        def _expand_distribution(self, dist, parent_plates, batch_shape=None):
            """
            Expand the distribution's batch shape to match the currently active
            plates in the context.
            """

            batch_shape = self._distribution_batch_shape(parent_plates, batch_shape=batch_shape)

            if not batch_shape or len(batch_shape) == 0:
                return dist
            else:
                return dist.expand(batch_shape + tuple(dist.batch_shape))
        
    class _BuildContext(_Context):
        """
        Context class for building the model, managing the state and operations
        during the model definition phase. It is responsible for tracing the sampled
        variables and their dependencies to construct the computational graph of the model.
        """

        def __init__(self, model, batch_shape=()):
            super().__init__(model, state={})

            self._batch_shape = batch_shape
            
        def _collect_parents(self, value, parents):
            """
            Recursively collect parent sites from a distribution's parameters.
            """

            if isinstance(value, Model._TraceTensor):
                parents.extend(value.parents)
                return
            if isinstance(value, dict):
                for item in value.values():
                    self._collect_parents(item, parents)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    self._collect_parents(item, parents)

        def _strip_trace_tensors(self, value):
            """
            Recursively strip trace tensors from a value and replace them with
            their raw tensor values.
            """

            if isinstance(value, Model._TraceTensor):
                return value.raw()
            if isinstance(value, dict):
                return {key: self._strip_trace_tensors(item) for key, item in value.items()}
            if isinstance(value, list):
                return [self._strip_trace_tensors(item) for item in value]
            if isinstance(value, tuple):
                return tuple(self._strip_trace_tensors(item) for item in value)
            return value

        def _sanitize_distribution(self, dist):
            """
            Sanitize a distribution by stripping trace tensors from its attributes and
            replacing them with their raw tensor values.
            """

            for key, value in list(dist.__dict__.items()):
                setattr(dist, key, self._strip_trace_tensors(value))
            return dist

        def _distribution_parents(self, dist):
            """
            Collect the parent sites of a distribution by inspecting its parameters.
            """

            parents = []
            for value in dist.__dict__.values():
                self._collect_parents(value, parents)
            return Model._TraceTensor._deduplicate_sites(parents)

        def _parent_value_extractor(self, value):
            """
            Return a function that can extract the current value of a distribution
            parameter from the state, given the original value which may contain trace
            tensors.
            """
            if isinstance(value, Model._TraceTensor):
                parent_site = getattr(value, "_trace_site", None)
                if parent_site is not None:
                    return lambda state, parent_site=parent_site: parent_site.value(state)

                raw_value = value.raw()
                return lambda state, raw_value=raw_value: raw_value

            if isinstance(value, list):
                extractors = [self._parent_value_extractor(item) for item in value]
                return lambda state, extractors=extractors: [extractor(state) for extractor in extractors]

            if isinstance(value, tuple):
                extractors = tuple(self._parent_value_extractor(item) for item in value)
                return lambda state, extractors=extractors: tuple(extractor(state) for extractor in extractors)

            if isinstance(value, dict):
                extractors = {key: self._parent_value_extractor(item) for key, item in value.items()}
                return lambda state, extractors=extractors: {key: extractor(state) for key, extractor in extractors.items()}

            return lambda state, value=value: value

        def sample(self, name, dist, observed=False):
            """
            This function is called when the model definition calls context.sample().
            It is used to trace the sampled variables and their dependencies during the
            build phase.
            """

            if name in self.model.sites:
                raise ValueError(f"Site '{name}' already exists in the model.")

            # Find the parents of the distribution by inspecting its parameters
            parents = self._distribution_parents(dist)

            # If any incoming edge from a parent site crosses plate boundaries, the new site
            # should be expanded to match the batch shape of the active plates. If the parents
            # have the same plate context as the new site, then no expansion is needed.
            if parents:
                # TODO: what if we go outside a plate? Is that possible?

                # Detect any plate boundary crossings by comparing parent plates and active plates.
                # Plates that the parent sites live in
                parent_plates = set(plate for parent in parents for plate in parent.plates)
                # Active plates that the current site is in
                active_plates = set(self._plate_stack)
                
                if not parent_plates.issubset(active_plates):
                    raise ValueError(f"Invalid dependency from site '{name}' to parent site(s) {[parent.name for parent in parents]} across plate boundaries.")

                if parent_plates != active_plates:
                    # This is not a root site, so it already inherits the batch shape via its parents
                    dist = self._expand_distribution(
                        dist,
                        parent_plates,
                        batch_shape=()
                    )
            else:
                # This is a root site so we need to expand it to match the batch shape
                dist = self._expand_distribution(
                    dist,
                    parent_plates = set(),
                    batch_shape = self._batch_shape
                )

            dist = self._sanitize_distribution(dist)
            
            if observed:
                site = Observed(name, dist, parents=parents, plates=list(self._plate_stack))
            else:
                site = Variable(name, dist, parents=parents, plates=list(self._plate_stack))

            for parent in parents:
                parent.children.append(site)
            
            self.model.sites[name] = site
            setattr(self.model, name, site)

            value = dist.sample()
            site.set(self.state, value)

            return Model._TraceTensor(value, site=site)
        
        def select(self, name, values, indices):
            if name in self.model.sites:
                raise ValueError(f"Site '{name}' already exists in the model.")

            # Collect input-value parents separately so we can register the
            # Selection site as a selector on each of them.
            input_parents = []
            self._collect_parents(values, input_parents)
            input_parents = Model._TraceTensor._deduplicate_sites(input_parents)

            # The selector is the single direct site behind the index expression.
            selector_site = getattr(indices, "_trace_site", None) if isinstance(indices, Model._TraceTensor) else None
            
            # The parents of the selection site include the input-value parents and the selector site (if any).
            index_parents = [selector_site] if selector_site is not None else []
            parents = Model._TraceTensor._deduplicate_sites(input_parents + index_parents)

            values_extractor = self._parent_value_extractor(values)
            indices_extractor = self._parent_value_extractor(indices)

            def eval_func(state):
                return torch.select(
                    values_extractor(state),
                    indices_extractor(state)
                )

            site = Selection(
                name,
                eval_func,
                parents=parents,
                plates=list(self._plate_stack),
                selector=selector_site,
            )

            for parent in parents:
                parent.children.append(site)

            # Register the Selection site as a selector on each input-value site
            # so that every site knows which selectors apply to it.
            for parent in input_parents:
                parent.selectors.append(site)

            self.model.sites[name] = site
            setattr(self.model, name, site)

            selected = torch.select(values, indices)
            if isinstance(selected, Model._TraceTensor):
                selected = selected.raw()

            # Evaluate and set value
            site.set(self.state)

            return Model._TraceTensor(selected, site=site)
        
        def step(
            self,
            name,
            sites, /,
            proposal = Constants.MISSING,
            propose_func = Constants.MISSING,
            update_func=Constants.MISSING,
            log_prob_func = Constants.MISSING,
        ):
            """
            Define a Gibbs sampling step for a block of sites.
            """

            if proposal is Constants.MISSING or not isinstance(proposal, Proposal):
                raise ValueError("Proposal must be an instance of the Proposal class.")
                    
            # If not provided, generate a default propose_func that samples from the proposal distribution
            # and assigns the proposed value to the step sites in the step state.
            if propose_func is Constants.MISSING:
                def propose_func(step, state):
                    sample = step.proposal.sample()
                    if len(step.sites) > 1:
                        for i, site in enumerate(step.sites):
                            site.set(state, sample[..., i])
                    else:
                        step.sites[0].set(state, sample)

            # If not provided, generate a default update_func that updates the proposal's
            # internal state based on the current values of the step sites in the step state.
            if update_func is Constants.MISSING:
                def update_func(step, state):
                    if len(step.sites) > 1:
                        sample = torch.stack([ site.value(state) for site in step.sites ], dim=-1)
                    else:
                        sample = step.sites[0].value(state)
                    step.proposal.update(sample)

            # Compute and persist factor-site metadata for this step. This is used by
            # the default log_prob implementation and made available for custom ones.
            step_site_set = set(sites)

            # Plates that the step sites already live in - used to detect
            # plate-boundary crossings when summing child log-probs.
            step_plates = set(p for s in sites for p in s.plates)

            # Select all factors that touch this step block, then evaluate their
            # corresponding stochastic site log-probabilities.
            factor_sites = set()
            for factor in self.model.factor_graph.factors:
                if any(site in step_site_set for site in factor.scope):
                    factor_sites.add(factor.site)

            # For each factor site, precompute the tensor dimensions (as
            # negative indices) that correspond to plates not present in the
            # step sites. Those dimensions must be summed out so that the
            # returned log-prob has the same shape as the step-site values.
            def _extra_plate_dims(factor_site):
                total_plate_dims = sum(len(p.size) for p in factor_site.plates)
                dims = []
                offset = 0

                for p in factor_site.plates:
                    n = len(p.size)

                    if p not in step_plates:
                        for k in range(n):
                            dims.append(-(total_plate_dims - offset - k))

                    offset += n

                # Batch dimensions are not summed out, so we need to shift the
                # negative indices by the number of batch dims.
                dims = [ dim - len(self._batch_shape) for dim in dims ]

                return tuple(dims)

            ordered_factor_sites = [
                fs
                for fs in self.model.sites.values()
                if fs in factor_sites and isinstance(fs, Variable)
            ]

            step_plate_dims = {
                fs: _extra_plate_dims(fs)
                for fs in ordered_factor_sites
            }

            # If not provided, generate a default log_prob_func that computes the
            # log-probability of the full conditional distribution for the step sites

            if log_prob_func is Constants.MISSING:
                def log_prob_func(step, state):
                    """
                    Evaluate the total conditional log-probability for the step sites
                    given the current state. This is computed by summing
                    the log-probabilities of all factors touching the step block,
                    taking into account
                    any extra plate dimensions that need to be summed out.
                    """

                    # Rebuild distributions from the current state so log_prob
                    # is evaluated with up-to-date parent-dependent parameters.
                    self.model.refresh(state)

                    total_log_prob = None

                    # If any of the edges that need to be evaluated here cross plate boundaries,
                    # we need to sum out the extra plate dimensions to get the correct shape for
                    # the log-probability.

                    for factor_site in step.factor_sites:
                        plate_dims = step.plate_dims[factor_site]
                        site_log_prob = factor_site.log_prob(state)

                        # If a site is used as an input to one or more Selection nodes,
                        # its factor contribution is gated by the corresponding selector
                        # variable(s). When the selector points to a different input,
                        # this site's factor should not contribute.
                        for selection in factor_site.selectors:
                            selector_site = selection.selector
                            if selector_site is None:
                                raise NotImplementedError("Selection sites must have a selector site.")

                            input_sites = [parent for parent in selection.parents if parent is not selector_site]
                            if factor_site not in input_sites:
                                raise NotImplementedError("Only direct parents of a selector site can be gated by it.")

                            gate_index = input_sites.index(factor_site)
                            gate_mask = selector_site.value(state) == gate_index
                            site_log_prob = torch.where(gate_mask, site_log_prob, torch.zeros_like(site_log_prob))

                        # Sum out any extra plate dimensions
                        if plate_dims:
                            site_log_prob = site_log_prob.sum(plate_dims)
                        
                        if total_log_prob is None:
                            total_log_prob = site_log_prob
                        else:
                            total_log_prob = total_log_prob + site_log_prob

                    return total_log_prob

            step = Step(
                name,
                sites,
                proposal = proposal,
                propose_func = propose_func,
                update_func = update_func,
                log_prob_func = log_prob_func,
                factor_sites = ordered_factor_sites,
                plate_dims = step_plate_dims,
            )

            self.model.steps[name] = step

    class _SampleContext(_Context):
        """
        Sample context used to execute the model definition and sample from the model.
        """

        def __init__(self, model, state=None, batch_shape=()):
            super().__init__(model, state=state)

            self.__batch_shape = batch_shape

        def __get_batch_shape(self):
            return self.__batch_shape

        batch_shape = property(__get_batch_shape)

        def sample(self, name, dist, observed=False):
            """
            Sample a value from the given distribution, optionally using an observed value.

            If the site is observed and a value is already present in the state, return the observed value.
            Otherwise, sample a new value from the distribution and store it in the state.
            """

            site = self.model.sites.get(name)
            is_root_site = site is None or len(site.parents) == 0
            batch_shape = self.__batch_shape if is_root_site else None
            parent_plates = set(plate for parent in site.parents for plate in parent.plates)
            
            dist = self._expand_distribution(dist, parent_plates, batch_shape=batch_shape)

            if observed and name in self.state:
                return self.state[name]

            value = dist.sample()
            site.set(self.state, value)
            return value

        def select(self, name, values, indices):
            """
            Select a value from the given list of tensors using the provided indices
            and store it in a new deterministic site.
            """

            site = self.model.sites.get(name)
            site.set(self.state)                # Evaluate the selection
            return site.value(self.state)

    class _RefreshContext(_Context):
        """
        Replay context used to refresh each site's distribution parameters from
        the current state without resampling values.
        """

        def __init__(self, model, state, batch_shape=()):
            super().__init__(model, state=state)

        def sample(self, name, dist, observed=False):
            site = self.model.sites.get(name)
            site.dist = dist                    # Update the distribution parameters
            return site.value(self.state)

        def select(self, name, values, indices):
            site = self.model.sites.get(name)
            return site.value(self.state)

    def __init__(self, dtype=Defaults.dtype):
        self.__dtype = dtype

        self.__batch_shape = ()
        self.__sites = OrderedDict()
        self.__plates = OrderedDict()
        self.__steps = OrderedDict()
        self.__factor_graph = None

    #region Properties

    def __get_dtype(self):
        return self.__dtype
    
    dtype = property(__get_dtype)

    def __get_sites(self):
        return self.__sites
    
    sites = property(__get_sites)

    def __get_plates(self):
        return self.__plates
    
    plates = property(__get_plates)

    def __get_steps(self):
        return self.__steps
    
    steps = property(__get_steps)

    def __get_factor_graph(self):
        return self.__factor_graph

    factor_graph = property(__get_factor_graph)

    #endregion

    def reset(self):
        self.__batch_shape = ()
        self.__sites.clear()
        self.__plates.clear()
        self.__steps.clear()
        self.__factor_graph = None

    def model(self, context):
        raise NotImplementedError("The 'model' method must be implemented by the subclass.")
    
    def build(self, batch_shape=()):
        """
        Build the model by executing the model definition with a build context to trace
        the variables and their dependencies. Then sample from the model with the given
        batch shape to initialize the proposals.
        """

        with torch.no_grad():
            # Trace the model variables and their dependencies to build the network
            build_context = Model._BuildContext(self, batch_shape=batch_shape)
            self.__batch_shape = batch_shape
            self.model(build_context)

            # Build a factor graph from the traced dependencies.
            self.__factor_graph = self.__build_factor_graph()

            # Call the step functions to allow proposals to initialize their internal
            # state based on the initial samples
            self.step(build_context)

        return build_context
    
    def sample(self, state=Constants.MISSING):
        state = state if state is not Constants.MISSING else {}

        if not self.sites:
            raise RuntimeError("Model has not been built yet. Call 'build()' before sampling.")

        sample_context = Model._SampleContext(self, state, batch_shape=self.__batch_shape)
        with torch.no_grad():
            self.model(sample_context)

        return state

    def refresh(self, state):
        if not self.sites:
            raise RuntimeError("Model has not been built yet. Call 'build()' before refreshing.")

        refresh_context = Model._RefreshContext(self, state, batch_shape=self.__batch_shape)
        with torch.no_grad():
            self.model(refresh_context)

        return state

    def __as_sites(self, sites):
        if hasattr(sites, "name") and hasattr(sites, "parents") and hasattr(sites, "children"):
            sites = [sites]
        else:
            sites = list(sites)

        if len(sites) == 0:
            raise ValueError("At least one site must be provided.")

        for current_site in sites:
            if not (hasattr(current_site, "name") and hasattr(current_site, "parents") and hasattr(current_site, "children")):
                raise ValueError("Invalid site reference. Use a Site object or an iterable of Site objects.")

        return sites

    def __build_factor_graph(self):
        """
        Construct a factor graph from the traced model.

        One factor is created for each stochastic site. Each factor scope includes
        the factor site itself and all stochastic ancestors that influence it through
        deterministic chains.
        """

        stochastic_sites = [
            site
            for site in self.sites.values()
            if isinstance(site, Variable)
        ]

        def stochastic_ancestors(site):
            result = set()
            visited = set()
            frontier = list(site.parents)

            while frontier:
                current = frontier.pop()
                current_id = id(current)

                if current_id in visited:
                    continue
                visited.add(current_id)

                if isinstance(current, Deterministic):
                    frontier.extend(current.parents)
                elif isinstance(current, Variable):
                    result.add(current)

            return result

        def selector_dependencies(site):
            """
            Collect stochastic selector variables that can gate the contribution
            of this site through downstream Selection nodes.
            """

            result = set()

            def upstream_stochastic_sites(node):
                deps = set()
                visited = set()
                frontier = [node]

                while frontier:
                    current = frontier.pop()
                    current_id = id(current)

                    if current_id in visited:
                        continue
                    visited.add(current_id)

                    if isinstance(current, Deterministic):
                        frontier.extend(current.parents)
                    elif isinstance(current, Variable):
                        deps.add(current)

                return deps

            for selection in site.selectors:
                selector_site = selection.selector
                if selector_site is None:
                    continue
                result.update(upstream_stochastic_sites(selector_site))

            return result

        factors = []
        for factor_site in stochastic_sites:
            deps = stochastic_ancestors(factor_site)
            deps.update(selector_dependencies(factor_site))
            scope = [
                site
                for site in stochastic_sites
                if site == factor_site or site in deps
            ]

            factors.append(
                Factor(
                    name=f"f_{factor_site.name}",
                    site=factor_site,
                    scope=scope,
                )
            )

        return FactorGraph(stochastic_sites, factors)

    def markov_blanket(self, sites, *, include_query_sites=False):
        """
        Return the Markov blanket for one site or a set of sites.

        For each query site X, MB(X) = parents(X) U children(X) U coparents(X),
        where deterministic chains directly connected to X are traversed to reach
        stochastic nodes:
        - parents: follow deterministic chains upward from X's direct parents
        - children: follow deterministic chains downward from X's direct children
        - coparents: for each stochastic child C, follow deterministic chains upward
          from C's direct parents

        Deterministic chains are only traversed when directly attached to the query
        node (or to a stochastic child). Traversal stops at stochastic boundaries.

        Parameters:
        -----------
        sites: Site or list of Site
            The site(s) for which to compute the Markov blanket.
        include_query_sites: bool, optional
            Whether to include the query site(s) themselves in the returned blanket. Default is False.
        """

        query_sites = self.__as_sites(sites)
        query_set = set(query_sites)
        blanket = set()
        selectors = set()

        def stochastic_ancestors(site):
            """
            Follow deterministic chains upward from site's parents to collect stochastic ancestors.
            Stops at stochastic nodes without traversing through them.
            """
            result = {}
            used_selections = set()
            visited = {}
            frontier = [ (parent, set()) for parent in site.parents ]
            while frontier:
                current, path_selections = frontier.pop()
                current_id = id(current)
                previous = visited.get(current_id)
                if previous is not None and path_selections.issubset(previous):
                    continue
                if previous is None:
                    visited[current_id] = set(path_selections)
                else:
                    previous.update(path_selections)
                if isinstance(current, Deterministic):
                    next_path = set(path_selections)
                    if isinstance(current, Selection):
                        used_selections.add(current)
                        next_path.add(current)
                    for parent in current.parents:
                        frontier.append((parent, next_path))
                else:
                    if current not in result:
                        result[current] = set(path_selections)
                    else:
                        result[current].update(path_selections)
            return result, used_selections

        def stochastic_children(site):
            """
            Follow deterministic chains downward from site's children to collect stochastic children.
            Stops at stochastic nodes without traversing through them.
            """
            result = {}
            used_selections = set()
            visited = {}
            frontier = [ (child, set()) for child in site.children ]
            while frontier:
                current, path_selections = frontier.pop()
                current_id = id(current)
                previous = visited.get(current_id)
                if previous is not None and path_selections.issubset(previous):
                    continue
                if previous is None:
                    visited[current_id] = set(path_selections)
                else:
                    previous.update(path_selections)
                if isinstance(current, Deterministic):
                    next_path = set(path_selections)
                    if isinstance(current, Selection):
                        used_selections.add(current)
                        next_path.add(current)
                    for child in current.children:
                        frontier.append((child, next_path))
                else:
                    if current not in result:
                        result[current] = set(path_selections)
                    else:
                        result[current].update(path_selections)
            return result, used_selections

        for query_site in query_sites:
            # Parents: follow deterministic chains upward from direct parents
            parents, parent_selectors = stochastic_ancestors(query_site)
            blanket.update(parents.keys())
            selectors.update(parent_selectors)

            # Children: follow deterministic chains downward from direct children
            children, child_selectors = stochastic_children(query_site)
            blanket.update(children.keys())
            selectors.update(child_selectors)

            # Co-parents: for each stochastic child, follow deterministic chains upward from its parents
            for child in children.keys():
                coparents, coparent_selectors = stochastic_ancestors(child)
                blanket.update(coparents.keys())
                selectors.update(coparent_selectors)

        if include_query_sites:
            blanket.update(query_set)
        else:
            blanket.difference_update(query_set)

        ordered_blanket = [ site for site in self.sites.values() if site in blanket ]
        ordered_selectors = [ site for site in self.sites.values() if site in selectors ]

        return Blanket(ordered_blanket, selections=ordered_selectors)


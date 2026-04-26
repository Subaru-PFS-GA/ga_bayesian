from collections import OrderedDict

import torch

from pfs.ga.bayesian import site

from .constants import Constants
from .defaults import Defaults
from .variable import Variable
from .observed import Observed
from .deterministic import Deterministic
from .proposal import Proposal
from .step import Step
from .plate import Plate

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

        def __init__(self, model):
            self.__model = model
            self._plate_stack = []

        #region Properties

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

        def __init__(self, model):
            super().__init__(model)
            self._deterministic_index = 0
            
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
                parent_plates = set(plate for parent in parents for plate in parent.plates)
                active_plates = set(self._plate_stack)
                
                if not parent_plates.issubset(active_plates):
                    raise ValueError(f"Invalid dependency from site '{name}' to parent site(s) {[parent.name for parent in parents]} across plate boundaries.")

                if parent_plates != active_plates:
                    dist = self._expand_distribution(dist, parent_plates)

            dist = self._sanitize_distribution(dist)
            
            if observed:
                site = Observed(name, dist, parents=parents, plates=list(self._plate_stack))
            else:
                site = Variable(name, dist, parents=parents, plates=list(self._plate_stack))

            for parent in parents:
                parent.children.append(site)
            
            self.model.sites[name] = site
            setattr(self.model, name, site)

            return Model._TraceTensor(dist.sample(), site=site)
        
        def select(self, name, values, indices):
            if name in self.model.sites:
                raise ValueError(f"Site '{name}' already exists in the model.")

            parents = []
            self._collect_parents(values, parents)
            self._collect_parents(indices, parents)
            parents = Model._TraceTensor._deduplicate_sites(parents)

            values_extractor = self._parent_value_extractor(values)
            indices_extractor = self._parent_value_extractor(indices)

            def eval_func(state):
                return torch.select(
                    values_extractor(state),
                    indices_extractor(state)
                )

            s = Deterministic(name, eval_func, parents=parents, plates=list(self._plate_stack))
            for parent in parents:
                parent.children.append(s)

            self.model.sites[name] = s
            setattr(self.model, name, s)

            selected = torch.select(values, indices)
            if isinstance(selected, Model._TraceTensor):
                selected = selected.raw()

            return Model._TraceTensor(selected, site=s)

    class _SampleContext(_Context):
        def __init__(self, model, state, batch_shape=()):
            super().__init__(model)
            self.__state = state
            self.__batch_shape = batch_shape

        #region Properties

        def __get_state(self):
            return self.__state

        state = property(__get_state)

        def __get_batch_shape(self):
            return self.__batch_shape

        batch_shape = property(__get_batch_shape)

        #endregion

        def sample(self, name, dist, observed=False):
            """
            Sample a value from the given distribution, optionally using an observed value.

            If the site is observed and a value is already present in the state, return the observed value.
            Otherwise, sample a new value from the distribution and store it in the state.
            """

            site = self.model.sites.get(name)
            is_root_site = site is None or len(site.parents) == 0
            batch_shape = self.batch_shape if is_root_site else None
            parent_plates = set(plate for parent in site.parents for plate in parent.plates)
            
            dist = self._expand_distribution(dist, parent_plates, batch_shape=batch_shape)

            if observed and name in self.state:
                return self.state[name]

            value = dist.sample()
            self.state[name] = value
            return value

        def select(self, name, values, indices):
            """
            Select a value from the given list of tensors using the provided indices
            and store it in a new deterministic site.
            """

            value = torch.select(values, indices)
            self.state[name] = value
            return value

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

            # If not provided, generate a default log_prob_func that computes the
            # log-probability of the full conditional distribution for the step sites

            if log_prob_func is Constants.MISSING:
                def log_prob_func(step, state):
                    # This is a placeholder for calculating the full conditional
                    pass

            step = Step(
                name,
                sites,
                proposal = proposal,
                propose_func = propose_func,
                update_func = update_func,
                log_prob_func = log_prob_func
            )

            self.model.steps[name] = step

    def __init__(self, dtype=Defaults.dtype):
        self.__dtype = dtype

        self.__sites = OrderedDict()
        self.__plates = OrderedDict()
        self.__steps = OrderedDict()

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

    #endregion

    def model(self, context):
        raise NotImplementedError("The 'model' method must be implemented by the subclass.")
    
    def sample(self, state=Constants.MISSING, batch_shape=()):
        if batch_shape is Constants.MISSING or batch_shape is None:
            batch_shape = ()
        elif isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        else:
            batch_shape = tuple(batch_shape)
        
        state = state if state is not Constants.MISSING else {}

        if not self.sites:
            build_context = Model._BuildContext(self)
            with torch.no_grad():
                self.model(build_context)

        sample_context = Model._SampleContext(self, state, batch_shape=batch_shape)
        with torch.no_grad():
            self.model(sample_context)

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

    def markov_blanket(self, sites, *, include_sites=False):
        """
        Return the Markov blanket for one site or a set of sites.

        The Markov blanket includes:
        - Stochastic parents of the query site(s)
        - All deterministic nodes that are connected to the query site(s)
          through a chain of deterministic nodes
        - All stochastic children of the query site(s)
        - All stochastic descendants of the query site(s) that are connected
          through a chain of deterministic nodes
        - All co-parents of the stochastic descendants described above
        - All deterministic descendants of the query site(s) that are
          connected to a stochastic descendant through a chain of deterministic nodes
        - All stochastic parents of the deterministic descendants described above

        Parameters:
        -----------
        sites: Site or list of Site
            The site(s) for which to compute the Markov blanket.
        include_sites: bool, optional
            Whether to include the query site(s) themselves in the returned blanket. Default is False.

        For each query site X, MB(X) = parents(X) U children(X) U parents(children(X)).
        When multiple query sites are provided, the result is the union of blankets.
        """

        query_sites = self.__as_sites(sites)
        query_set = set(query_sites)

        blanket = set()

        def deterministic_closure(seed_sites):
            """
            Compute the closure of deterministic nodes reachable from the seed sites.

            This is used to find all deterministic ancestors and descendants that can
            connect stochastic nodes in the Markov blanket.
            """
            closure = set()
            frontier = list(seed_sites)

            while frontier:
                current_det = frontier.pop()
                if current_det in closure:
                    continue
                closure.add(current_det)

                for parent in current_det.parents:
                    if isinstance(parent, Deterministic):
                        frontier.append(parent)

                for child in current_det.children:
                    if isinstance(child, Deterministic):
                        frontier.append(child)

            return closure

        for query_site in query_sites:
            blanket.update(query_site.parents)

            # Deterministic parents can hide additional random ancestors.
            det_parents = [ parent for parent in query_site.parents if isinstance(parent, Deterministic) ]
            det_parent_closure = deterministic_closure(det_parents)
            for det_site in det_parent_closure:
                blanket.add(det_site)
                blanket.update(det_site.parents)

            for child in query_site.children:
                if isinstance(child, Deterministic):
                    # Traverse deterministic chain in both directions and include
                    # first stochastic descendants and their co-parents.
                    det_child_closure = deterministic_closure([child])
                    for det_site in det_child_closure:
                        blanket.add(det_site)
                        blanket.update(det_site.parents)

                        for det_child in det_site.children:
                            if not isinstance(det_child, Deterministic):
                                blanket.add(det_child)
                                blanket.update(det_child.parents)
                else:
                    blanket.add(child)
                    blanket.update(child.parents)

                    # If a stochastic child feeds deterministic transforms, those
                    # deterministic nodes and their parents are part of the blanket,
                    # but stochastic descendants beyond that are not.
                    det_grandchildren = [grandchild for grandchild in child.children if isinstance(grandchild, Deterministic)]
                    det_grandchild_closure = deterministic_closure(det_grandchildren)
                    for det_site in det_grandchild_closure:
                        blanket.add(det_site)
                        blanket.update(det_site.parents)

        if include_sites:
            blanket.update(query_set)
        else:
            blanket.difference_update(query_set)

        ordered_blanket = [ site for site in self.sites.values() if site in blanket ]
        return ordered_blanket

    def log_prob_markov_blanket(self, state, sites, *, include_sites=True):
        """
        Sum log-probability terms over the Markov blanket of the given site(s).

        Deterministic sites are skipped because they do not define a probability term.
        """

        blanket_sites = self.markov_blanket(sites, include_sites=include_sites)

        total_log_prob = None
        for blanket_site in blanket_sites:
            if not hasattr(blanket_site, "log_prob"):
                continue

            site_log_prob = blanket_site.log_prob(state)
            if total_log_prob is None:
                total_log_prob = site_log_prob
            else:
                total_log_prob = total_log_prob + site_log_prob

        if total_log_prob is None:
            query_site = self.__as_sites(sites)[0]
            value = query_site.value(state)
            return torch.zeros_like(value, dtype=value.dtype)

        return total_log_prob

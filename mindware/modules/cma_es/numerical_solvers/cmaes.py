import cma
from mindware.modules.cma_es.numerical_solvers.base import NumericalSolverBase
from typing import List
import numpy as np
import time

class CMAES(NumericalSolverBase):
    """Numerical Solver CMA-ES to find a weight vector (ensemble weighting).

    Super Parameters
    ----------
        see NumericalSolverBase for more details on args and kwargs

    --- Method Parameters
   batch_size: , default=25
        The batch size of CMA-ES ("popsize" for CMAES).
    """

    def __init__(self, *args, batch_size=25, bounded: bool = False, sigma0=False, time_limit=3600, **kwargs) -> None:

        if isinstance(batch_size, int):
            tmp_batch_size = batch_size
        elif batch_size == "dynamic":
            # Following CMA-ES default
            tmp_batch_size = int(4 + 3 * np.log(len(args[0])))
        else:
            raise ValueError(f"Unknown batch size argument! Got: {batch_size}")

        super().__init__(*args, **kwargs, batch_size=tmp_batch_size)
        self.bounded = bounded
        self.time_limit = time_limit

        # FIXME: did this due to config space (None not supported; and otherwise names not unique)
        if not sigma0:
            self.sigma0 = 0.2
        else:
            self.sigma0 = sigma0

    def _compute_internal_iterations(self):
        # -- Determine iteration handling
        n_evals = self.n_evaluations - self.n_init_evaluations
        internal_n_iterations = n_evals // self.batch_size
        if n_evals % self.batch_size == 0:
            n_rest_evaluations = 0
        else:
            n_rest_evaluations = n_evals % self.batch_size

        return internal_n_iterations, n_rest_evaluations

    def _minimize(self, predictions: List[np.ndarray], labels: np.ndarray, _start_weight_vector: np.ndarray):

        # internal_n_iterations, n_rest_evaluations = self._compute_internal_iterations()
        es = self._setup_cma(_start_weight_vector)
        val_loss_over_iterations = []

        start = time.time()
        while time.time() - start <= self.time_limit:
            # Ask/tell
            solutions = es.ask()
            es.tell(solutions,
                    self._evaluate_batch_of_solutions(solutions, predictions, labels))
            es.disp(modulo=1)

            # Iteration finalization
            val_loss_over_iterations.append(es.result.fbest)            

        # Iterations
        # for itr in range(1, internal_n_iterations + 1):
        #     # Ask/tell
        #     solutions = es.ask()
        #     es.tell(solutions,
        #             self._evaluate_batch_of_solutions(solutions, predictions, labels))
        #     es.disp(modulo=1)

        #     # Iteration finalization
        #     val_loss_over_iterations.append(es.result.fbest)

        #     # -- ask/tell rest solutions
        # if n_rest_evaluations > 0:
        #     solutions = es.ask(n_rest_evaluations)
        #     es.best.update(solutions,
        #                    arf=self._evaluate_batch_of_solutions(solutions, predictions, labels))

        #     print("Evaluated {} rest solutions in a remainder iteration.".format(n_rest_evaluations))
        #     val_loss_over_iterations.append(es.result.fbest)

        return es.result.fbest, es.result.xbest, val_loss_over_iterations

    def _setup_cma(self, _start_weight_vector) -> cma.CMAEvolutionStrategy:

        # Setup CMA
        opts = cma.CMAOptions()
        opts.set("seed", self.random_state.randint(0, 1000000))
        opts.set("popsize", self.batch_size)
        # opts.set("maxfevals", self.remaining_evaluations_)  # Not used because we control by hand.

        if self.bounded:
            opts.set("bounds", [0, 1])

        sigma0 = self.sigma0

        es = cma.CMAEvolutionStrategy(_start_weight_vector, sigma0, inopts=opts)

        return es

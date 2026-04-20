/* Minimal OpenMP stub for the MATLAB Coder watershed sources.
 *
 * The codegen C uses `omp_nest_lock_t`, `omp_init_nest_lock`,
 * `omp_destroy_nest_lock`, `omp_set_nest_lock`, `omp_unset_nest_lock`, and
 * `omp_get_max_threads`, plus `#pragma omp parallel for`. We don't need
 * the parallelism (our outer pipeline already runs segments in parallel),
 * so we stub these to no-ops and let the compiler ignore the pragma
 * when -fopenmp is not passed.
 *
 * This lets us build a self-contained extension that doesn't depend on
 * libomp at load time.
 */
#ifndef OMP_STUB_H
#define OMP_STUB_H

#ifdef __cplusplus
extern "C" {
#endif

typedef int omp_nest_lock_t;

static inline void omp_init_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_destroy_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_set_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline void omp_unset_nest_lock(omp_nest_lock_t *lock) { (void)lock; }
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }

#ifdef __cplusplus
}
#endif

#endif /* OMP_STUB_H */

/**
 * topo_node.h — Topological Pointer-Reuse Network Engine
 *
 * Core data structures for the DAS-based congruence algebra network.
 * Replaces dense tensor computation with discrete topology graph traversal,
 * pointer reuse, and p-adic precision expansion.
 *
 * Key concepts:
 *   - TopoNode: A concept node storing discrete congruence state (no floats)
 *   - Directed edges as pointer arrays (morphisms in the DAS category)
 *   - Observer-relative propagation from a chosen origin
 *   - Taylor-decay truncation for natural far-field cutoff
 *   - Leftward precision expansion on semantic collision
 */

#ifndef H2Q_TOPO_NODE_H
#define H2Q_TOPO_NODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ───── Configuration constants ───── */
#define TOPO_MAX_EDGES        64
#define TOPO_MAX_PRECISION    32
#define TOPO_DEFAULT_PRECISION 4

/* ───── Core node structure ───── */

/**
 * TopoNode — A single concept in the topology network.
 *
 * Instead of storing float weight vectors, each node stores:
 *   - base_id: integer token / concept identity
 *   - p_adic_precision: leftward congruence precision depth
 *   - directed_edges: pointer array to downstream context nodes (morphisms)
 *   - num_edges: current fan-out count
 *   - relative_step_distance: distance from current observer origin
 *   - congruence_class: residue class under current modulus
 *   - visited: BFS / propagation flag
 */
typedef struct TopoNode {
    uint64_t base_id;
    uint32_t p_adic_precision;
    uint32_t congruence_class;   /* base_id mod (2^p_adic_precision) */

    struct TopoNode **directed_edges;
    uint16_t num_edges;
    uint16_t edge_capacity;

    uint16_t relative_step_distance;
    uint8_t  visited;
    uint8_t  _pad;
} TopoNode;

/* ───── Propagation statistics ───── */

typedef struct PropagationStats {
    uint64_t morphism_count;       /* total pointer jumps */
    uint64_t nodes_visited;
    uint64_t truncation_events;    /* far-field Taylor cutoffs */
    uint64_t collision_events;     /* semantic collisions detected */
    uint32_t max_step_reached;
    uint32_t max_precision_seen;
    double   elapsed_us;           /* wall-clock microseconds */
} PropagationStats;

/* ───── Benchmark result ───── */

typedef struct BenchmarkResult {
    uint64_t num_nodes;
    uint64_t num_edges;
    double   build_time_us;
    double   propagate_time_us;
    uint64_t morphism_count;
    uint64_t truncation_events;
    uint64_t collision_events;
    uint32_t max_step;
    double   memory_bytes;
    double   ops_per_second;
} BenchmarkResult;

/* ───── Lifecycle ───── */

TopoNode *topo_node_create(uint64_t base_id, uint32_t precision);
void      topo_node_destroy(TopoNode *node);

/* ───── Edge management ───── */

int  topo_node_add_edge(TopoNode *from, TopoNode *to);
void topo_node_remove_edge(TopoNode *from, uint16_t index);

/* ───── Congruence algebra ───── */

/** Recompute congruence_class = base_id mod 2^precision */
void topo_node_update_congruence(TopoNode *node);

/** Check whether two nodes collide (same congruence class at their precision) */
int  topo_node_collides(const TopoNode *a, const TopoNode *b);

/* ───── Observer-relative propagation ───── */

/**
 * Propagate from `origin` up to `max_steps` hops, applying Taylor-decay
 * truncation.  Fills `stats` with metrics.
 */
void propagate_from_origin(TopoNode *origin, int max_steps,
                           PropagationStats *stats);

/* ───── Taylor decay ───── */

/**
 * Returns the Taylor-decay weight for a given step:
 *   decay = 1.0 / step_s!
 * Truncation occurs when decay < 1.0 / 2^precision_level.
 */
double calculate_taylor_decay(int step_s, int precision_level);

/* ───── Leftward precision expansion ───── */

/**
 * On semantic collision, split `collision_node` into two children with
 * higher precision.  Returns pointer to the first child (the second is
 * linked as its first directed edge).
 *
 * All incoming edges that previously pointed to `collision_node` must be
 * rebound by the caller.
 */
TopoNode *expand_leftward_precision(TopoNode *collision_node);

/* ───── Network builder (for benchmarks) ───── */

/**
 * Build a random topology network with `n` nodes and average `avg_edges`
 * edges per node.  Returns a flat array of node pointers.
 * Caller must free with `topo_network_destroy`.
 */
TopoNode **topo_network_build(uint64_t n, uint16_t avg_edges, uint32_t seed);
void       topo_network_destroy(TopoNode **nodes, uint64_t n);

/** Reset visited flags on all nodes */
void topo_network_reset(TopoNode **nodes, uint64_t n);

/* ───── Full benchmark entry point ───── */

BenchmarkResult topo_run_benchmark(uint64_t num_nodes, uint16_t avg_edges,
                                   int max_steps, uint32_t seed);

#ifdef __cplusplus
}
#endif

#endif /* H2Q_TOPO_NODE_H */

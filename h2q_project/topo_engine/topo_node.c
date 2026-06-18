/**
 * topo_node.c — Topological Pointer-Reuse Network Engine (Implementation)
 *
 * Pure C implementation of the DAS-based congruence algebra network.
 * All "computation" is discrete pointer traversal + integer arithmetic.
 * No floating-point weight matrices anywhere.
 */

#define _POSIX_C_SOURCE 199309L

#include "topo_node.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define TOPO_INITIAL_EDGE_CAPACITY 8u
#define TOPO_MIN_QUEUE_CAPACITY    2u
#define TOPO_BFS_QUEUE_CAPACITY    65536u
#define TOPO_BUILD_MIN_PRECISION   2u
#define TOPO_BUILD_PRECISION_RANGE 7u   /* generates 2..8 inclusive */

/* ────────────────────────────────────────────────────────────────────
 * Portable high-resolution timer (POSIX)
 * ──────────────────────────────────────────────────────────────────── */

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

/* ────────────────────────────────────────────────────────────────────
 * Lifecycle
 * ──────────────────────────────────────────────────────────────────── */

TopoNode *topo_node_create(uint64_t base_id, uint32_t precision) {
    TopoNode *n = (TopoNode *)calloc(1, sizeof(TopoNode));
    if (!n) return NULL;

    n->base_id          = base_id;
    n->p_adic_precision = (precision > 0) ? precision : TOPO_DEFAULT_PRECISION;
    n->edge_capacity    = TOPO_INITIAL_EDGE_CAPACITY;  /* start small, grow on demand */
    n->directed_edges   = (TopoNode **)calloc(n->edge_capacity, sizeof(TopoNode *));
    n->num_edges        = 0;
    n->visited          = 0;

    topo_node_update_congruence(n);
    return n;
}

void topo_node_destroy(TopoNode *node) {
    if (!node) return;
    free(node->directed_edges);
    free(node);
}

/* ────────────────────────────────────────────────────────────────────
 * Edge management
 * ──────────────────────────────────────────────────────────────────── */

int topo_node_add_edge(TopoNode *from, TopoNode *to) {
    if (!from || !to) return -1;
    if (from->num_edges >= TOPO_MAX_EDGES) return -2;

    /* Grow if needed */
    if (from->num_edges >= from->edge_capacity) {
        uint16_t new_cap = from->edge_capacity * 2;
        if (new_cap > TOPO_MAX_EDGES) new_cap = TOPO_MAX_EDGES;
        TopoNode **tmp = (TopoNode **)realloc(
            from->directed_edges, new_cap * sizeof(TopoNode *));
        if (!tmp) return -3;
        from->directed_edges = tmp;
        from->edge_capacity  = new_cap;
    }

    from->directed_edges[from->num_edges++] = to;
    return 0;
}

void topo_node_remove_edge(TopoNode *from, uint16_t index) {
    if (!from || index >= from->num_edges) return;
    /* Shift left */
    for (uint16_t i = index; i + 1 < from->num_edges; i++) {
        from->directed_edges[i] = from->directed_edges[i + 1];
    }
    from->num_edges--;
}

/* ────────────────────────────────────────────────────────────────────
 * Congruence algebra
 * ──────────────────────────────────────────────────────────────────── */

void topo_node_update_congruence(TopoNode *node) {
    if (!node) return;
    uint32_t modulus = 1u << node->p_adic_precision;
    node->congruence_class = (uint32_t)(node->base_id % modulus);
}

int topo_node_collides(const TopoNode *a, const TopoNode *b) {
    if (!a || !b) return 0;
    /* Use the minimum precision of both nodes */
    uint32_t min_prec = (a->p_adic_precision < b->p_adic_precision)
                        ? a->p_adic_precision : b->p_adic_precision;
    uint32_t modulus = 1u << min_prec;
    return ((a->base_id % modulus) == (b->base_id % modulus))
           && (a->base_id != b->base_id);  /* same class but different identity */
}

/* ────────────────────────────────────────────────────────────────────
 * Taylor decay
 * ──────────────────────────────────────────────────────────────────── */

/* Table holds 0! through 20!.  20! ≈ 2.4e18 fits in double without
   precision loss.  calculate_taylor_decay returns 0.0 for step > 20
   because 1/21! is below any practical threshold. */
static double factorial_table[21];
static int    factorial_ready = 0;

static void init_factorial(void) {
    if (factorial_ready) return;
    factorial_table[0] = 1.0;
    for (int i = 1; i <= 20; i++) {
        factorial_table[i] = factorial_table[i - 1] * (double)i;
    }
    factorial_ready = 1;
}

double calculate_taylor_decay(int step_s, int precision_level) {
    init_factorial();
    (void)precision_level;  /* reserved for future threshold logic */
    if (step_s < 0) step_s = 0;
    if (step_s > 20) return 0.0;  /* essentially zero */

    double decay = 1.0 / factorial_table[step_s];

    /* Note: this function only returns the raw decay value.
       The truncation threshold (1 / 2^precision) is applied by the
       caller (propagate_from_origin) to decide whether to prune. */
    return decay;
}

/* ────────────────────────────────────────────────────────────────────
 * Observer-relative propagation (BFS with Taylor truncation)
 * ──────────────────────────────────────────────────────────────────── */

/* Simple ring-buffer queue for BFS */
typedef struct {
    TopoNode **buf;
    uint64_t   cap;
    uint64_t   head;
    uint64_t   tail;
} BFSQueue;

static BFSQueue *bfs_queue_create(uint64_t cap) {
    if (cap < TOPO_MIN_QUEUE_CAPACITY) cap = TOPO_MIN_QUEUE_CAPACITY;
    BFSQueue *q = (BFSQueue *)malloc(sizeof(BFSQueue));
    if (!q) return NULL;
    q->buf  = (TopoNode **)malloc(cap * sizeof(TopoNode *));
    if (!q->buf) {
        free(q);
        return NULL;
    }
    q->cap  = cap;
    q->head = 0;
    q->tail = 0;
    return q;
}

static void bfs_queue_destroy(BFSQueue *q) {
    if (!q) return;
    free(q->buf);
    free(q);
}

static int bfs_queue_push(BFSQueue *q, TopoNode *n) {
    uint64_t next = (q->tail + 1) % q->cap;
    if (next == q->head) return -1;  /* full */
    q->buf[q->tail] = n;
    q->tail = next;
    return 0;
}

static TopoNode *bfs_queue_pop(BFSQueue *q) {
    if (q->head == q->tail) return NULL;  /* empty */
    TopoNode *n = q->buf[q->head];
    q->head = (q->head + 1) % q->cap;
    return n;
}

static int bfs_queue_empty(const BFSQueue *q) {
    return q->head == q->tail;
}

void propagate_from_origin(TopoNode *origin, int max_steps,
                           PropagationStats *stats)
{
    init_factorial();
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));

    if (!origin) return;

    double t0 = now_us();

    /* Start BFS from origin */
    BFSQueue *queue = bfs_queue_create(TOPO_BFS_QUEUE_CAPACITY);
    if (!queue) return;

    origin->visited = 1;
    origin->relative_step_distance = 0;
    if (bfs_queue_push(queue, origin) != 0) {
        stats->queue_overflow_events++;
        bfs_queue_destroy(queue);
        return;
    }
    stats->nodes_visited = 1;

    while (!bfs_queue_empty(queue)) {
        TopoNode *cur = bfs_queue_pop(queue);
        uint16_t cur_step = cur->relative_step_distance;

        if ((int)cur_step >= max_steps) continue;

        /* Taylor decay threshold for this node's precision */
        double threshold = 1.0 / (double)(1u << cur->p_adic_precision);

        for (uint16_t i = 0; i < cur->num_edges; i++) {
            TopoNode *child = cur->directed_edges[i];
            if (!child || child->visited) continue;

            uint16_t child_step = cur_step + 1;
            stats->morphism_count++;

            /* ── Taylor truncation check ── */
            double decay = calculate_taylor_decay((int)child_step,
                                                  (int)cur->p_adic_precision);
            if (decay < threshold) {
                stats->truncation_events++;
                continue;  /* prune: too far */
            }

            /* ── Semantic collision check ── */
            if (topo_node_collides(cur, child)) {
                stats->collision_events++;
                /* In a full system we'd call expand_leftward_precision here */
            }

            if (bfs_queue_push(queue, child) != 0) {
                stats->queue_overflow_events++;
                continue;
            }

            child->visited = 1;
            child->relative_step_distance = child_step;
            stats->nodes_visited++;

            if (child_step > stats->max_step_reached)
                stats->max_step_reached = child_step;
            if (child->p_adic_precision > stats->max_precision_seen)
                stats->max_precision_seen = child->p_adic_precision;
        }
    }

    bfs_queue_destroy(queue);
    stats->elapsed_us = now_us() - t0;
}

/* ────────────────────────────────────────────────────────────────────
 * Leftward precision expansion
 * ──────────────────────────────────────────────────────────────────── */

TopoNode *expand_leftward_precision(TopoNode *collision_node) {
    if (!collision_node) return NULL;
    if (collision_node->p_adic_precision >= TOPO_MAX_PRECISION) return NULL;

    uint32_t new_prec = collision_node->p_adic_precision + 1;
    uint64_t precision_delta = 1ULL << collision_node->p_adic_precision;
    if (UINT64_MAX - collision_node->base_id < precision_delta) return NULL;

    /* Create two children with higher precision */
    TopoNode *child_a = topo_node_create(collision_node->base_id, new_prec);
    TopoNode *child_b = topo_node_create(
        collision_node->base_id + precision_delta,
        new_prec);

    if (!child_a || !child_b) {
        topo_node_destroy(child_a);
        topo_node_destroy(child_b);
        return NULL;
    }

    /* Inherit edges from collision node to both children */
    for (uint16_t i = 0; i < collision_node->num_edges; i++) {
        topo_node_add_edge(child_a, collision_node->directed_edges[i]);
        topo_node_add_edge(child_b, collision_node->directed_edges[i]);
    }

    /* Link children: child_a -> child_b as its first directed edge */
    topo_node_add_edge(child_a, child_b);

    return child_a;
}

/* ────────────────────────────────────────────────────────────────────
 * Network builder (for benchmarks)
 * ──────────────────────────────────────────────────────────────────── */

/* Simple LCG PRNG for reproducibility */
static uint64_t lcg_state;
static void     lcg_seed(uint32_t s) { lcg_state = s; }
static uint64_t lcg_next(void) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return lcg_state >> 16;
}

TopoNode **topo_network_build(uint64_t n, uint16_t avg_edges, uint32_t seed) {
    if (n == 0) return NULL;
    lcg_seed(seed);

    TopoNode **nodes = (TopoNode **)calloc(n, sizeof(TopoNode *));
    if (!nodes) return NULL;

    /* Create nodes with random precision 2..8 */
    for (uint64_t i = 0; i < n; i++) {
        uint32_t prec = TOPO_BUILD_MIN_PRECISION
                      + (uint32_t)(lcg_next() % TOPO_BUILD_PRECISION_RANGE);
        nodes[i] = topo_node_create(i, prec);
        if (!nodes[i]) {
            topo_network_destroy(nodes, i);
            return NULL;
        }
    }

    /* Add random directed edges */
    for (uint64_t i = 0; i < n; i++) {
        uint16_t ne = (uint16_t)(1 + lcg_next() % (2 * avg_edges));
        if (ne > TOPO_MAX_EDGES) ne = TOPO_MAX_EDGES;
        for (uint16_t e = 0; e < ne; e++) {
            uint64_t target = lcg_next() % n;
            if (target != i) {
                topo_node_add_edge(nodes[i], nodes[target]);
            }
        }
    }

    return nodes;
}

void topo_network_destroy(TopoNode **nodes, uint64_t n) {
    if (!nodes) return;
    for (uint64_t i = 0; i < n; i++) {
        topo_node_destroy(nodes[i]);
    }
    free(nodes);
}

void topo_network_reset(TopoNode **nodes, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        if (nodes[i]) {
            nodes[i]->visited = 0;
            nodes[i]->relative_step_distance = 0;
        }
    }
}

/* ────────────────────────────────────────────────────────────────────
 * Full benchmark entry point
 * ──────────────────────────────────────────────────────────────────── */

BenchmarkResult topo_run_benchmark(uint64_t num_nodes, uint16_t avg_edges,
                                   int max_steps, uint32_t seed)
{
    BenchmarkResult result;
    memset(&result, 0, sizeof(result));

    result.num_nodes = num_nodes;

    /* Build network */
    double t0 = now_us();
    TopoNode **network = topo_network_build(num_nodes, avg_edges, seed);
    double t1 = now_us();
    result.build_time_us = t1 - t0;

    if (!network) return result;

    /* Count total edges */
    for (uint64_t i = 0; i < num_nodes; i++) {
        result.num_edges += network[i]->num_edges;
    }

    /* Propagate from node 0 */
    PropagationStats pstats;
    topo_network_reset(network, num_nodes);

    double t2 = now_us();
    propagate_from_origin(network[0], max_steps, &pstats);
    double t3 = now_us();

    result.propagate_time_us = t3 - t2;
    result.morphism_count    = pstats.morphism_count;
    result.truncation_events = pstats.truncation_events;
    result.collision_events  = pstats.collision_events;
    result.queue_overflow_events = pstats.queue_overflow_events;
    result.max_step          = pstats.max_step_reached;

    /* Memory estimate: nodes + edge pointers */
    result.memory_bytes = (double)num_nodes * sizeof(TopoNode)
                        + (double)result.num_edges * sizeof(TopoNode *);

    if (result.propagate_time_us > 0) {
        result.ops_per_second = (double)pstats.morphism_count
                              / (result.propagate_time_us / 1e6);
    }

    topo_network_destroy(network, num_nodes);
    return result;
}

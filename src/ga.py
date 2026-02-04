import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

# =============================================================================
# CORE IMPLEMENTATION ADDITIONS (P5 Mario)
# =============================================================================
# All modified spots (exactly 4). Each has "Changes made:" and "What it does:".
# Ctrl + F to find the changes made.
#   1. Config + mutation weights
#   2. Individual_Grid.mutate()
#   3. Individual_Grid.generate_children()
#   4. _select_parent() + generate_successors()
# =============================================================================
# Locations and behavior:
#
# 1. Config + Mutation Weights
#    - Constants control mutation rate, crossover mode, elitism, and selection.
#    - Why: So we can switch strategies (e.g. roulette vs tournament) and tune
#      the GA without editing logic. Weighted tiles bias mutation toward empty
#      space and coins instead of uniform random, which keeps levels playable.
#
# 2. Individual_Grid.mutate()
#    - Per-gene mutation: each interior tile (non-floor, non-border) is replaced
#      with probability MUTATION_RATE_GENE using the weighted tile list.
#    - Why: Per-gene rate gives fine-grained exploration; skipping floor/border
#      keeps the level structure valid (mario start, flag, solid floor).
#
# 3. Individual_Grid.generate_children()
#    - Crossover: uniform (50/50 per cell), single-point (one cut in row-major),
#      or multi-point (alternating segments). Then mutate the child genome.
#    - Why: Crossover mixes parent levels; uniform is unbiased, single-point
#      preserves contiguous regions, multi-point allows more mixing. Mutation
#      after crossover adds exploration.
#
# 4. _select_parent() and generate_successors()
#    - Selection: roulette (fitness-proportionate), tournament (best of K), or
#      random. Roulette uses shifted fitness so negative fitnesses become valid
#      weights. generate_successors keeps top ELITE_COUNT, then fills the rest
#      by selecting two parents and adding children until population size is met.
#    - Why: Elitism prevents losing the best solution. Tournament is robust to
#      fitness scale; roulette emphasizes better individuals; random is a
#      baseline for diversity.
# =============================================================================

width = 200
height = 16

# --- CORE IMPLEMENTATION (1): GA configuration ---

# ===========================================================================================
# Changes made: Added constants for mutation rate, crossover mode/points, elite count,
#   selection strategy, and tournament size; added weighted tile list for mutation.

# What it does: Central place to tune the GA. Mutation weights favor "-" and "o" over
#   pipes/enemies so levels stay playable; other constants choose selection and crossover.
# ===========================================================================================

MUTATION_RATE_GENE = 0.02           # Per-gene mutation probability (per tile)
CROSSOVER_MODE = "uniform"          # "uniform" | "single_point" | "multi_point"
CROSSOVER_POINTS = 3                # Number of cut points for multi-point crossover
ELITE_COUNT = 48                    # Top-k individuals carried unchanged (e.g. 10% of 480)
SELECTION_STRATEGY = "tournament"   # "roulette" | "tournament" | "random"
TOURNAMENT_SIZE = 5                 # K for tournament selection (pick best of K random)

# Tile weights for mutation (higher = more likely). Ensures more empty/coins, fewer pipes/enemies.
_mutation_tile_weights = [
    ("-", 12), ("X", 4), ("?", 2), ("M", 1), ("B", 2), ("o", 6), ("|", 1), ("T", 1), ("E", 2)
]
_mutation_tiles, _mutation_weights = zip(*_mutation_tile_weights)

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f", # a flag, do not generate
    #"v", # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

#
# The level as a grid of tiles
#
class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # --- CORE IMPLEMENTATION (2): Mutation ---

    # ===========================================================================================
    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    # ===========================================================================================
    # Changes made: Replaced empty loop with per-gene mutation; use MUTATION_RATE_GENE and
    #   weighted tile list (_mutation_tiles / _mutation_weights); only touch interior, non-floor cells.

    # What it does: For each mutable tile (rows 0..height-2, cols 1..width-2), with probability
    #   MUTATION_RATE_GENE replaces it with a new tile chosen by weight (more empty/coins, fewer pipes).
    # ===========================================================================================

    def mutate(self, genome):
        left, right = 1, width - 1
        for y in range(height - 1):  # skip last row (floor)
            for x in range(left, right):
                if random.random() < MUTATION_RATE_GENE:
                    genome[y][x] = random.choices(_mutation_tiles, weights=_mutation_weights, k=1)[0]
        return genome

    # --- CORE IMPLEMENTATION (3): Crossover ---

    # ===========================================================================================
    # Create zero or more children from self and other (crossover + mutation).
    # ===========================================================================================
    # Changes made: Replaced empty crossover with three modes: uniform (50/50 per cell),
    #   single-point (one cut in row-major order), multi-point (alternating segments); then call mutate.

    # What it does: Builds one child genome by combining self and other per CROSSOVER_MODE, mutates it,
    #   and returns a single Individual_Grid. Floor row and first/last columns are never crossed.
    # ===========================================================================================

    def generate_children(self, other):
        left, right = 1, width - 1
        rows_used = height - 1  # exclude floor row for crossover
        total_cells = rows_used * (right - left)

        if CROSSOVER_MODE == "uniform":
            new_genome = copy.deepcopy(self.genome)
            for y in range(rows_used):
                for x in range(left, right):
                    if random.random() < 0.5:
                        new_genome[y][x] = other.genome[y][x]
        elif CROSSOVER_MODE == "single_point":
            cut = random.randint(0, total_cells) if total_cells > 0 else 0
            new_genome = copy.deepcopy(self.genome)
            idx = 0
            for y in range(rows_used):
                for x in range(left, right):
                    if idx >= cut:
                        new_genome[y][x] = other.genome[y][x]
                    idx += 1
        else:  # multi_point
            new_genome = copy.deepcopy(self.genome)
            n_pts = min(CROSSOVER_POINTS, max(0, total_cells - 1))
            if n_pts > 0 and total_cells > 1:
                cuts = sorted([0] + random.sample(range(1, total_cells), n_pts) + [total_cells])
                for seg_i in range(len(cuts) - 1):
                    use_other = (seg_i % 2) == 1
                    for idx in range(cuts[seg_i], cuts[seg_i + 1]):
                        y = idx // (right - left)
                        x = left + idx % (right - left)
                        if use_other:
                            new_genome[y][x] = other.genome[y][x]
        self.mutate(new_genome)
        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_Grid

# --- CORE IMPLEMENTATION (4): Selection + generate_successors ---

# ===========================================================================================
# Changes made: Added _select_parent() and full generate_successors(). Selection supports
#   roulette (shifted fitness so negative values work), tournament (best of K), and random.
#   generate_successors keeps top ELITE_COUNT, then fills rest by selecting two parents and
#   adding children until population size is reached.

# What it does: _select_parent returns one individual for breeding. generate_successors
#   returns the next generation: elite individuals plus offspring from selected parent pairs.
# ===========================================================================================

def _select_parent(population, strategy=SELECTION_STRATEGY):
    """Select one individual: roulette (fitness-proportionate), tournament (best of K), or random."""
    if not population:
        return None
    if strategy == "random":
        return random.choice(population)
    if strategy == "tournament":
        k = min(TOURNAMENT_SIZE, len(population))
        pool = random.sample(population, k)
        return max(pool, key=Individual.fitness)
    # roulette: fitness-proportionate (shift fitnesses to be non-negative)
    fitnesses = [Individual.fitness(p) for p in population]
    min_f = min(fitnesses)
    weights = [f - min_f + 1e-6 for f in fitnesses]
    total = sum(weights)
    if total <= 0:
        return random.choice(population)
    probs = [w / total for w in weights]
    return random.choices(population, weights=probs, k=1)[0]


def generate_successors(population):
    """Build next generation: elitism (top ELITE_COUNT) + offspring via selection and crossover."""
    pop_limit = len(population)
    # Sort best first for elitism
    sorted_pop = sorted(population, key=Individual.fitness, reverse=True)
    results = list(sorted_pop[:ELITE_COUNT])
    need = pop_limit - len(results)
    for _ in range(need):
        p1 = _select_parent(population)
        p2 = _select_parent(population)
        if p1 is None or p2 is None:
            break
        children = p1.generate_children(p2)
        results.extend(children)
        if len(results) >= pop_limit:
            break
    return results[:pop_limit]


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")

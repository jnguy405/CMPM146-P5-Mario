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
#   1. Config + mutation weights
#   2. Individual_Grid.calculate_fitness()
#   3. Individual_Grid.mutate() + Individual_Grid.random_individual()
#   4. Individual_Grid.generate_children() + _select_parent() + generate_successors()
#
# Citations:
# - Fitness function metrics inspired by various Mario level generation papers and the provided metrics.py.
# - Mutation and random generation ideas inspired by common Mario level design patterns and structures.
# - Some comments generated with Co-Pilot to explain the rationale behind design choices.
# =============================================================================
# Locations and behavior:
#
# 1. Config + Mutation Weights
#    - Added GA configuration constants (MUTATION_RATE_GENE, ELITE_COUNT, etc.)
#      and weighted tile list for mutation.
#    - Why: Centralized configuration allows easy tuning. Weighted tiles favor
#      empty space and coins over pipes/enemies, producing more playable levels
#      by default.
#
# 2. Individual_Grid.calculate_fitness()
#    - Enhanced fitness function with playability metrics: path continuity,
#      enemy placement checks, coin clustering, and penalties for bad design
#      (floating pipes, unreachable items, too many enemies).
#    - Why: Original fitness was arbitrary. New metrics makes sure levels have
#      continuous paths, properly placed enemies, and logical item placement
#      while penalizing common level design flaws.
#
# 3. Individual_Grid.mutate() + Individual_Grid.random_individual()
#    - mutate(): Per-gene mutation using weighted tiles, skipping floor/border.
#    - random_individual(): Structured level generation with sections (platforms,
#      pipes, coins, enemies, mixed) and strict pipe placement on ground.
#    - Why: Mutation explores while maintaining structure. Random generation
#      creates Mario-like levels with coherent sections instead of random noise.
#      Pipes are strictly placed on ground to prevent floating pipes.
#
# 4. Individual_Grid.generate_children() + _select_parent() + generate_successors()
#    - generate_children(): Section-based crossover (4 sections with 50% chance
#      to take from other parent per section).
#    - _select_parent(): Tournament selection (default), roulette, or random.
#    - generate_successors(): Elitism (top ELITE_COUNT) + offspring generation.
#    - Why: Section crossover preserves large structural elements. Tournament
#      selection balances exploration/exploitation. Elitism preserves best
#      solutions while generating diverse offspring.
# =============================================================================

width = 200
height = 16

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
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# Tile weights for mutation (higher = more likely). Gives more empty/coins, fewer pipes/enemies.
_mutation_tile_weights = [
    ("-", 6),    # Empty space
    ("X", 8),    # Solid blocks (common)
    ("?", 3),    # Question blocks
    ("M", 1),    # Mushroom blocks (rare)
    ("B", 4),    # Breakable blocks
    ("o", 5),    # Coins
    ("|", 1),    # Pipe segment (rare - needs proper placement)
    ("T", 1),    # Pipe top (rare - needs proper placement)
    ("E", 2),    # Enemies
]
_mutation_tiles, _mutation_weights = zip(*_mutation_tile_weights)

# GA Configuration
MUTATION_RATE_GENE = 0.01
ELITE_COUNT = 48
SELECTION_STRATEGY = "tournament"
TOURNAMENT_SIZE = 7

# The level as a grid of tiles
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
        
        level = self.to_level()
        
        # 1. Check for reasonable enemy placement (not in pits)
        enemy_placement = 0
        enemy_count = 0
        enemy_well_placed = 0
        for y in range(height):
            for x in range(width):
                if level[y][x] == "E":
                    enemy_count += 1
                    # Enemy should have ground below (within reasonable distance)
                    ground_found = False
                    for check_y in range(y + 1, min(y + 3, height)):
                        if level[check_y][x] in ["X", "B", "?", "M"]:
                            ground_found = True
                            break
                    if ground_found:
                        enemy_well_placed += 1
        
        if enemy_count > 0:
            enemy_placement = enemy_well_placed / enemy_count
        
        # 2. STRICT PIPE CHECK: Pipes must be on ground
        pipe_penalty = 0
        for y in range(height):
            for x in range(width):
                if level[y][x] == "T":  # Pipe top
                    # Pipe must have ground below it
                    pipe_height = 1
                    pipe_valid = True
                    
                    # Check pipe body
                    check_y = y + 1
                    while check_y < height and level[check_y][x] in ["|", "T"]:
                        pipe_height += 1
                        check_y += 1
                    
                    # Pipe bottom should be on ground (row 15 has "X")
                    if check_y >= height or level[check_y][x] != "X":
                        pipe_valid = False
                        pipe_penalty += 2  # Strong penalty for floating pipes
                    
                    # Bonus for properly placed pipes
                    if pipe_valid:
                        pipe_penalty -= 0.5  # Small reward for good pipes
        
        # 3. Check for clustered coins (better than random scatter)
        coin_clustering = 0
        coin_positions = []
        for y in range(height):
            for x in range(width):
                if level[y][x] == "o":
                    coin_positions.append((x, y))
        
        if len(coin_positions) > 1:
            distances = []
            for i in range(len(coin_positions)):
                for j in range(i + 1, len(coin_positions)):
                    dist = abs(coin_positions[i][0] - coin_positions[j][0])
                    if dist < 5:  # Coins close together
                        distances.append(1.0)
                    elif dist < 10:
                        distances.append(0.5)
            if distances:
                coin_clustering = sum(distances) / len(distances)
        
        # Metrics
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,  # More tolerant of empty space
            pathPercentage=1.0,  
            emptyPercentage=0.6,
            linearity=0.3,
            solvability=2.0,  # Still important but not overriding
        )
        
        # Calculate weighted sum
        fitness = 0
        for metric, value in measurements.items():
            if metric in coefficients:
                fitness += coefficients[metric] * value
        
        # Metric contributions
        fitness += enemy_placement * 1.0
        fitness += coin_clustering * 1.0  # More reward for coin clusters
        
        # Only strict penalty: Bad pipe placement
        fitness -= pipe_penalty * 1.0
        
        # Penalties (keep them light)
        # Too many enemies (but be generous)
        if enemy_count > 15:
            fitness -= (enemy_count - 15) * 0.2
        
        # Too many pipes (but allow some)
        pipe_count = sum(row.count("T") for row in level)
        if pipe_count > 8:
            fitness -= (pipe_count - 8) * 0.2
        
        # Reward for having enemies and coins (encourage them!)
        if enemy_count > 0:
            fitness += min(enemy_count, 10) * 0.1
        
        coin_count = sum(row.count("o") for row in level)
        if coin_count > 0:
            fitness += min(coin_count, 30) * 0.05
        
        # Reward for question blocks and breakable blocks
        qblock_count = sum(row.count("?") + row.count("M") for row in level)
        fitness += min(qblock_count, 10) * 0.1
        
        breakable_count = sum(row.count("B") for row in level)
        fitness += min(breakable_count, 10) * 0.05
        
        self._fitness = fitness
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        left, right = 1, width - 1
        for y in range(height - 1):  # skip last row (floor)
            for x in range(left, right):
                if random.random() < MUTATION_RATE_GENE:
                    genome[y][x] = random.choices(_mutation_tiles, weights=_mutation_weights, k=1)[0]
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        
        # Keep section-based crossover
        sections = 4
        section_width = (width - 2) // sections
        
        for section in range(sections):
            start_x = 1 + section * section_width
            end_x = min(start_x + section_width, width - 2)
            
            if random.random() < 0.5:  # 50% chance to take from other parent for this section
                for y in range(height - 1):  # Skip floor
                    for x in range(start_x, end_x):
                        # STUDENT Which one should you take?  Self, or other?  Why?
                        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                        new_genome[y][x] = other.genome[y][x]
        
        # do mutation; note we're returning a one-element tuple here
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
        # Create a Mario-like level with proper structure
        g = [["-" for col in range(width)] for row in range(height)]
        
        # Solid floor
        g[15][:] = ["X"] * width
        
        # Mario start position
        g[14][0] = "m"
        
        # Flag and flagpole
        g[7][-1] = "v"
        for row in range(8, 14):
            g[row][-1] = "f"
        g[14][-1] = "X"
        g[15][-1] = "X"
        
        # Create structured sections
        sections = 5  # Number of distinct platforming sections
        section_width = (width - 2) // sections
        
        for section in range(sections):
            start_x = 1 + section * section_width
            end_x = min(start_x + section_width, width - 2)
            
            # Choose section type
            section_type = random.choice(["platforms", "pipes", "coins", "enemies", "mixed"])
            
            if section_type == "platforms":
                # Create floating platforms
                num_platforms = random.randint(1, 3)
                for _ in range(num_platforms):
                    plat_x = random.randint(start_x, end_x - 4)
                    plat_y = random.randint(10, 13)
                    plat_width = random.randint(3, 6)
                    plat_height = random.randint(0, 2)  # Multi-level platforms
                    
                    for y_offset in range(plat_height + 1):
                        for x_offset in range(plat_width):
                            if 0 <= plat_y - y_offset < height and 0 <= plat_x + x_offset < width:
                                g[plat_y - y_offset][plat_x + x_offset] = "X"
            
            elif section_type == "pipes":
                # Create pipes (ALWAYS ON GROUND - row 14 is ground level, row 15 is solid floor)
                num_pipes = random.randint(1, 2)
                for _ in range(num_pipes):
                    pipe_x = random.randint(start_x, end_x - 2)
                    pipe_height = random.randint(2, 4)
                    
                    # PIPE STRICTNESS: MUST be on ground
                    # Pipe top (row indices: 0 is top, 15 is bottom)
                    # Ground level is row 14 (just above solid floor at row 15)
                    # So pipe top should be at: 15 - pipe_height
                    g[15 - pipe_height][pipe_x] = "T"
                    # Pipe body
                    for h in range(1, pipe_height):
                        g[15 - pipe_height + h][pipe_x] = "|"
                    # Pipe base is already on ground (row 15 has "X")
            
            elif section_type == "coins":
                # Create coin patterns
                pattern = random.choice(["line", "arc", "block", "scatter", "staircase"])
                coin_x = random.randint(start_x, end_x - 5)
                coin_y = random.randint(10, 13)
                
                if pattern == "line":
                    for i in range(5):
                        if 0 <= coin_x + i < width:
                            g[coin_y][coin_x + i] = "o"
                elif pattern == "arc":
                    for i in range(5):
                        if 0 <= coin_x + i < width and 0 <= coin_y - abs(2 - i) < height:
                            g[coin_y - abs(2 - i)][coin_x + i] = "o"
                elif pattern == "block":
                    for i in range(3):
                        for j in range(3):
                            if 0 <= coin_y + i < height and 0 <= coin_x + j < width:
                                g[coin_y + i][coin_x + j] = "o"
                elif pattern == "scatter":
                    # Random scatter of coins in the section
                    for _ in range(random.randint(3, 8)):
                        scatter_x = random.randint(start_x, end_x - 1)
                        scatter_y = random.randint(8, 13)
                        if g[scatter_y][scatter_x] == "-":
                            g[scatter_y][scatter_x] = "o"
                elif pattern == "staircase":
                    # Diagonal staircase of coins
                    for i in range(5):
                        if 0 <= coin_x + i < width and 0 <= coin_y - i < height:
                            g[coin_y - i][coin_x + i] = "o"
            
            elif section_type == "enemies":
                # Place enemies - more variety
                num_enemies = random.randint(1, 4)  # More enemies possible
                for _ in range(num_enemies):
                    enemy_x = random.randint(start_x, end_x - 1)
                    
                    # Choose enemy type placement
                    enemy_type = random.choice(["ground", "platform", "floating"])
                    
                    if enemy_type == "ground":
                        # Enemy on ground
                        if g[15][enemy_x] == "X":
                            g[14][enemy_x] = "E"
                    elif enemy_type == "platform":
                        # Enemy on existing platform or new small platform
                        # Check if there's already a platform
                        placed = False
                        for y in range(11, 14):
                            if g[y][enemy_x] == "X":
                                g[y-1][enemy_x] = "E"
                                placed = True
                                break
                        if not placed:
                            # Create a small platform for the enemy
                            g[12][enemy_x] = "X"
                            g[13][enemy_x] = "E"
                    elif enemy_type == "floating":
                        # Enemy on a floating brick (sometimes unreachable for challenge)
                        if random.random() < 0.3:  # 30% chance for floating enemy
                            float_y = random.randint(8, 11)
                            if g[float_y][enemy_x] == "-":
                                g[float_y][enemy_x] = "E"
            
            elif section_type == "mixed":
                # Combination of elements - more generous with items
                # Add some question blocks
                num_qblocks = random.randint(1, 3)
                for _ in range(num_qblocks):
                    qblock_x = random.randint(start_x, end_x - 2)
                    qblock_y = random.randint(10, 12)
                    if g[qblock_y][qblock_x] == "-":
                        g[qblock_y][qblock_x] = "?" if random.random() < 0.7 else "M"
                
                # Add breakable blocks
                num_breakable = random.randint(1, 3)
                for _ in range(num_breakable):
                    block_x = random.randint(start_x, end_x - 1)
                    block_y = random.randint(11, 13)
                    if g[block_y][block_x] == "-":
                        g[block_y][block_x] = "B"
                
                # Maybe add some coins too
                if random.random() < 0.5:
                    coin_x = random.randint(start_x, end_x - 3)
                    coin_y = random.randint(10, 12)
                    for i in range(3):
                        if g[coin_y][coin_x + i] == "-":
                            g[coin_y][coin_x + i] = "o"
        
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
            pathPercentage=0.8,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        
        # Calculate base fitness first
        fitness = sum(map(lambda m: coefficients[m] * measurements[m], coefficients))
        
        # Pipe reward
        pipe_positions = [de[0] for de in self.genome if de[1] == "7_pipe"]
        if len(pipe_positions) > 1:
            pipe_proximity = sum(abs(a-b) for a,b in zip(pipe_positions, pipe_positions[1:]))
            fitness += max(0, 5.0 - pipe_proximity/50)

        penalties = 0
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        
        # Element count balancing
        element_counts = {}
        for de in self.genome:
            de_type = de[1]
            element_counts[de_type] = element_counts.get(de_type, 0) + 1
        
        # Penalize too many pipes
        if element_counts.get("7_pipe", 0) > 6:
            penalties -= (element_counts["7_pipe"] - 6) * 0.5
        
        # Penalize too many enemies
        if element_counts.get("2_enemy", 0) > 12:
            penalties -= (element_counts["2_enemy"] - 12) * 0.3
        
        # Reward good number of coins (8-20 is ideal)
        coin_count = element_counts.get("3_coin", 0)
        if 8 <= coin_count <= 20:
            fitness += 3.0
        elif coin_count > 0:
            fitness += min(coin_count / 20, 2.0)  # Partial reward
            
        self._fitness = fitness + penalties
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
                    new_x = offset_by_upto(x, width/15, min=1, max=width-2)
                    new_de = (new_x, de_type)
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        # STUDENT How does this work?  Explain it in your writeup.
        
        if len(self.genome) == 0:
            pa = 0
        else:
            pa = random.randint(0, len(self.genome) - 1)
        
        if len(other.genome) == 0:
            pb = 0
        else:
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
        # Instead of completely empty, give it a few basic elements
        g = []
        # Add at least one element to avoid empty genome issues
        g.append((width // 2, "1_platform", 4, 10, "X"))
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = []
        for i in range(elt_count):
            x = random.randint(1, width - 2)  # Avoid edges
            
            element_type = random.choice([
                "0_hole", "1_platform", "2_enemy", "3_coin", 
                "4_block", "5_qblock", "6_stairs", "7_pipe"
            ])
            
            if element_type == "0_hole":
                w = random.randint(1, 8)
                g.append((x, element_type, w))
            elif element_type == "1_platform":
                w = random.randint(1, 8)
                y = random.randint(0, height - 3)  # Keep platform above ground
                madeof = random.choice(["?", "X", "B"])
                g.append((x, element_type, w, y, madeof))
            elif element_type == "2_enemy":
                g.append((x, element_type))
            elif element_type == "3_coin":
                y = random.randint(0, height - 2)  # Keep coin above ground
                g.append((x, element_type, y))
            elif element_type == "4_block":
                y = random.randint(0, height - 2)  # Keep block above ground
                breakable = random.choice([True, False])
                g.append((x, element_type, y, breakable))
            elif element_type == "5_qblock":
                y = random.randint(0, height - 2)  # Keep qblock above ground
                has_powerup = random.choice([True, False])
                g.append((x, element_type, y, has_powerup))
            elif element_type == "6_stairs":
                h = random.randint(1, min(8, height - 4))  # Keep stairs reasonable height
                dx = random.choice([-1, 1])
                g.append((x, element_type, h, dx))
            elif element_type == "7_pipe":
                h = random.randint(2, height - 4)  # Ensure pipe fits within bounds
                g.append((x, element_type, h))
        
        return Individual_DE(g)


Individual = Individual_DE


def _select_parent(population, strategy=SELECTION_STRATEGY):
    """Select one individual: roulette (fitness-proportionate), tournament (best of K), or random."""
    if not population or len(population) == 0:
        return None
    if strategy == "random":
        return random.choice(population)
    if strategy == "tournament":
        k = min(TOURNAMENT_SIZE, len(population))
        if k == 0:
            return None
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
    results = []
    pop_limit = len(population)
    
    # Sort best first for elitism
    sorted_pop = sorted(population, key=Individual.fitness, reverse=True)
    results = list(sorted_pop[:ELITE_COUNT])
    need = pop_limit - len(results)
    
    for i in range(need):
        p1 = _select_parent(population)
        p2 = _select_parent(population)
        if p1 is None or p2 is None:
            # If we can't select parents, add a random individual
            results.append(Individual.random_individual())
            continue
            
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
    
    try:
        with mpool.Pool(processes=os.cpu_count()) as pool:
            init_time = time.time()
            # STUDENT (Optional) change population initialization
            print(f"Creating initial population of size {pop_limit}...")
            population = []
            for i in range(pop_limit):
                if random.random() < 0.9:
                    ind = Individual.random_individual()
                else:
                    ind = Individual.empty_individual()
                population.append(ind)
                if i % 100 == 0:  # Progress indicator
                    print(f"Created {i+1}/{pop_limit} individuals...")
            
            print(f"Successfully created {len(population)} individuals")
            print("Calculating initial fitnesses...")
            
            population = pool.map(Individual.calculate_fitness,
                                  population,
                                  batch_size)
            init_done = time.time()
            print(f"Created and calculated initial population statistics in:", init_done - init_time, "seconds")
            print(f"Population size after fitness calculation: {len(population)}")
            generation = 0
            start = time.time()
            now = start
            print("Use ctrl-c to terminate this loop manually.")
            best_fitness_history = []
            convergence_window = 10
            max_generations = 500
            
            # Track best individual across all generations
            all_time_best = None
            all_time_best_fitness = float('-inf')
            
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
                    
                    # Update all-time best
                    if best.fitness() > all_time_best_fitness:
                        all_time_best = best
                        all_time_best_fitness = best.fitness()
                        # Save all-time best immediately
                        timestamp = time.strftime("%m_%d_%H_%M_%S")
                        filename = f"levels/best_so_far_gen_{generation}_{timestamp}.txt"
                        with open(filename, 'w') as f:
                            for row in best.to_level():
                                f.write("".join(row) + "\n")
                        print(f"New all-time best! Saved to {filename}")
                
                generation += 1
                
                # Store best fitness for convergence check
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    best_fitness_history.append(best.fitness())
                    
                    # Check convergence
                    if generation > convergence_window:
                        recent_avg = sum(best_fitness_history[-convergence_window:]) / convergence_window
                        overall_avg = sum(best_fitness_history) / len(best_fitness_history)
                        if recent_avg > overall_avg * 0.95:  # Less than 5% improvement
                            print(f"Converged at generation {generation}")
                            break
                    
                    # Max generations
                    if generation >= max_generations:
                        print(f"Reached max generations: {max_generations}")
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
        print("\nInterrupted by user. Saving current best...")
        # Save current population before returning
        if population and len(population) > 0:
            best = max(population, key=Individual.fitness)
            timestamp = time.strftime("%m_%d_%H_%M_%S")
            filename = f"levels/interrupted_gen_{generation}_{timestamp}.txt"
            with open(filename, 'w') as f:
                for row in best.to_level():
                    f.write("".join(row) + "\n")
            print(f"Saved interrupted generation best: {filename}")
            
            # Also save all-time best if we have it
            if all_time_best:
                filename = f"levels/all_time_best_{timestamp}.txt"
                with open(filename, 'w') as f:
                    for row in all_time_best.to_level():
                        f.write("".join(row) + "\n")
                print(f"Saved all-time best: {filename}")
    
    print(f"\nGA terminated at generation {generation}")
    
    if population and len(population) > 0:
        best = max(population, key=Individual.fitness)
        print(f"Best fitness in final population: {best.fitness()}")
    else:
        print("WARNING: Population is empty!")
    
    return population  # Return whatever population we have (even if interrupted)


if __name__ == "__main__":
    # Create levels directory if it doesn't exist
    if not os.path.exists("levels"):
        os.makedirs("levels")
        print("Created 'levels' directory")
    
    # Run GA - it will return population even if interrupted
    try:
        final_population = ga()
    except Exception as e:
        print(f"Error during GA execution: {e}")
        final_population = []
    
    # Always save files, even if population is incomplete
    if final_population and len(final_population) > 0:
        final_gen = sorted(final_population, key=Individual.fitness, reverse=True)
        best = final_gen[0]
        print("Best fitness: " + str(best.fitness()))
        now = time.strftime("%m_%d_%H_%M_%S")
        
        # Save the top 10 levels (or fewer if we have less than 10)
        num_to_save = min(10, len(final_gen))
        print(f"Saving top {num_to_save} levels...")
        
        for k in range(0, num_to_save):
            filename = f"levels/{now}_top_{k+1}.txt"
            with open(filename, 'w') as f:
                for row in final_gen[k].to_level():
                    f.write("".join(row) + "\n")
            print(f"Saved: {filename} (fitness: {final_gen[k].fitness():.2f})")
    else:
        print("No levels to save. Population is empty or GA didn't run properly.")
        
        # Check if last.txt exists from previous run
        if os.path.exists("levels/last.txt"):
            print("Found 'levels/last.txt' from previous run")
            # Copy it with a timestamp
            timestamp = time.strftime("%m_%d_%H_%M_%S")
            backup_name = f"levels/backup_last_{timestamp}.txt"
            shutil.copy2("levels/last.txt", backup_name)
            print(f"Backed up last.txt as {backup_name}")
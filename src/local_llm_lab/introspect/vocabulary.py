"""The concepts, in families, split into train and held out.

Families exist so the mismatch negatives can be stratified: a same-family distractor is a harder
negative than a different-family one, and reporting them together hides which is being learned.
No word here appears in `vectors.BASELINE_POOL`, and none is one of the five the base-model grids
used -- those five stay clean so the two records can be compared without the training set having
touched them.
"""

from __future__ import annotations

FAMILIES: dict[str, tuple[str, ...]] = {
    "foods": ("bread", "cheese", "soup", "rice", "pasta", "apples", "honey", "pepper", "garlic",
              "chocolate", "porridge", "dumplings", "custard", "olives", "mustard", "cinnamon",
              "pickles", "yoghurt", "almonds", "sausages", "marmalade", "noodles", "vinegar",
              "biscuits", "salmon", "lentils", "walnuts", "figs", "toast", "broth"),
    "nature": ("the ocean", "mountains", "forests", "deserts", "glaciers", "volcanoes", "rivers",
               "swamps", "caves", "meadows", "coral reefs", "tundra", "waterfalls", "canyons",
               "prairies", "estuaries", "dunes", "wetlands", "fjords", "geysers", "monsoons",
               "avalanches", "earthquakes", "thunderstorms", "eclipses", "tides", "auroras",
               "fossils", "sediment", "erosion"),
    "animals": ("elephants", "sparrows", "octopuses", "wolves", "honeybees", "tortoises",
                "dolphins", "ravens", "moths", "salamanders", "pelicans", "badgers", "lemurs",
                "jellyfish", "hedgehogs", "falcons", "otters", "termites", "chameleons",
                "narwhals", "magpies", "ferrets", "starfish", "gibbons", "herons", "beetles",
                "puffins", "manatees", "cormorants", "aardvarks"),
    "places": ("Paris", "Reykjavik", "Cairo", "Kyoto", "Lisbon", "Havana", "Marrakesh",
               "Helsinki", "Valparaiso", "Samarkand", "Dubrovnik", "Antarctica", "Patagonia",
               "Siberia", "the Sahara", "the Alps", "Venice", "Istanbul", "Nairobi", "Seoul",
               "Quebec", "Tasmania", "Madagascar", "Greenland", "Bhutan", "Jerusalem", "Prague",
               "Casablanca", "Tangier", "Ushuaia"),
    "artefacts": ("lighthouses", "clocks", "telescopes", "bridges", "windmills", "submarines",
                  "cathedrals", "printing presses", "violins", "microscopes", "aqueducts",
                  "sundials", "typewriters", "kilns", "looms", "locomotives", "periscopes",
                  "astrolabes", "harpsichords", "catapults", "lanterns", "mosaics", "frescoes",
                  "tapestries", "obelisks", "sextants", "hourglasses", "barometers", "anvils",
                  "abacuses"),
    "abstractions": ("betrayal", "nostalgia", "justice", "patience", "envy", "courage", "grief",
                     "curiosity", "loyalty", "regret", "ambition", "humility", "boredom",
                     "wonder", "resentment", "gratitude", "shame", "defiance", "solitude",
                     "hope", "suspicion", "forgiveness", "dread", "delight", "restraint",
                     "obsession", "indifference", "yearning", "vigilance", "contempt"),
    "sciences": ("particle physics", "thermodynamics", "genetics", "astronomy", "geology",
                 "immunology", "cryptography", "topology", "linguistics", "metallurgy",
                 "seismology", "epidemiology", "optics", "botany", "hydrology", "acoustics",
                 "cartography", "meteorology", "palaeontology", "electrochemistry", "ecology",
                 "neurology", "statistics", "mineralogy", "entomology", "aerodynamics",
                 "radiology", "virology", "oceanography", "spectroscopy"),
    "activities": ("sailing", "beekeeping", "weaving", "bookbinding", "glassblowing", "fencing",
                   "gardening", "birdwatching", "pottery", "mountaineering", "calligraphy",
                   "surveying", "falconry", "distilling", "carpentry", "cartwheeling",
                   "orienteering", "foraging", "blacksmithing", "quilting", "juggling",
                   "spelunking", "whittling", "thatching", "tanning", "cooperage", "milling",
                   "dyeing", "engraving", "fermenting"),
}

#: Never trained on. The five the base-model grids measured, plus one per family held back so a
#: held-out result can be read family by family. A concept the training set has touched cannot say
#: anything about generalisation, and these five in particular join two records.
LEGACY = ("bread", "the ocean", "Paris", "betrayal", "butt holes")


def split(seed: int = 0, held_out_per_family: int = 6) -> tuple[dict, dict]:
    """(train, held_out), both {concept: family}. Deterministic given the seed."""
    import random
    rng = random.Random(seed)
    train: dict[str, str] = {}
    held: dict[str, str] = {}
    for family, words in FAMILIES.items():
        pool = [w for w in words if w not in LEGACY]
        rng.shuffle(pool)
        for word in pool[:held_out_per_family]:
            held[word] = family
        for word in pool[held_out_per_family:]:
            train[word] = family
    for word in LEGACY:
        for family, words in FAMILIES.items():
            if word in words:
                held[word] = family
    return train, held


def all_concepts() -> dict[str, str]:
    return {word: family for family, words in FAMILIES.items() for word in words}

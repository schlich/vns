from datatree import DataTree
from vns import extinction_learning


def test_souza_2020_fig_3a():
    """Return a dataframe to populate a figure similar to Souza et al. (2020) Fig. 3a."""
    datatree = DataTree()
    assert extinction_learning(datatree).dims == (
        "Day",
        "Condition",
        "Trial",
        "% Freezing",
    )

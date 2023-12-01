# metablox

metablox (metadata block structure exploration) is a Python library for quantifying the relationship between 
categorical node metadata and block structure of a network, as introduced in

[Mangold, Lena and Camille Roth. “Quantifying metadata-structure relationships in networks using 
description length.” (2023) arXiv:2311.18705 [cs.SI]](https://arxiv.org/abs/2311.18705).

Please cite the paper if you use the measure in your work.

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/lenafm/metablox.git
```

Then `cd` into the created `metablox` directory and use `pip` to install the package locally:

``` 
pip install -e .
```

## Usage

To calculate the metablox vector $`\gamma`$ for a network with one or multiple sets of metadata, use the
`calculate_gamma` function. Here, we show this on a co-purchasing network of political books[^1], which have a 
set of metadata of political leaning of the books (left, neutral, conservative), saved in a vertex property 
called `value`.

```python
import graph_tool.all as gt
from metablox.gamma import calculate_gamma

g = gt.collection.ns["polbooks"]
metadata = ['value']

gamma = calculate_gamma(g=g, metadata=metadata)
```

If you have a metadata partition for a network which you have not yet saved as a vertex property to the `graph-tool` 
Graph object, you can also pass the `metadata` as a list of `np.arrays`!

The `calculate_gamma` function returns a dictionary, with an entry for each element in the `metadata` list. For
each metadata, it includes a nested dictionary with the elements of the $`\gamma`$ vector, with the key being
the SBM variant and the value being the value of $`\gamma`$ for this variant.

For a simple plot of two dimensions of the $`\gamma`$ vector, simply call `plot_gamma`:
```python
import graph_tool.all as gt
from metablox.gamma import calculate_gamma
from metablox.plot import plot_gamma

g = gt.collection.ns["polbooks"]
metadata = ['value']
gamma = calculate_gamma(g=g, metadata=metadata)

plot_gamma(gamma=gamma, dimx='dc', dimy='pp')
```

For `dimx` and `dimy`, enter the SBM variant of the element of $`\gamma`$ you want to plot on the x and y-axes, 
respectively.

By default, `plot_gamma` plots all sets of metadata for which there is an entry in $`\gamma`$. You can pass a 
list of names of sets of metadata as `metadata` to only plot a subset of the metadata sets.

You can add annotations to the dots on the plot that represent the set of metadata (this is useful when you have
calculated $`\gamma`$ for a network that has multiple sets of metadata). You can also add a title:

```python
import graph_tool.all as gt
from metablox.gamma import calculate_gamma
from metablox.plot import plot_gamma

g = gt.collection.ns["polbooks"]
metadata = ['value']
gamma = calculate_gamma(g=g, metadata=metadata)

plot_gamma(gamma=gamma, dimx='dc', dimy='pp', 
           annotations={'value': 'Political leaning'}, 
           title='Co-purchasing book network')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)

[^1]: V. Krebs, "The political books network", unpublished, https://doi.org/10.2307/40124305
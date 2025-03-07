# IMDB Sentiment Analaysis Example

*   The data is downloaded from tensorflow_dataset.
*   The example dataset imdb_small_with_labels.csv that comes within /data is
    the first 100 entries of the original dataset.
*   See [imdb_fetch_data.py](/tfx/examples/imdb/imdb_fetch_data.py) to use the entire imdb dataset.
*   And please adjust the corresponding hyperparameters to account for the
    larger dataset.


# Data Source Acknowledgement

```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```

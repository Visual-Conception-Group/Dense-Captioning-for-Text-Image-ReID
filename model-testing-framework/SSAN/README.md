# Semantically Self-Aligned Network for Text-to-Image Part-aware Person Re-identification

After creating the neccessary dataset structures and preprocessing. 

#### Training and Testing
~~~
sh experiments/Dataset-Name/train.sh 
sh experiments/Dataset-Name/train.sh 
~~~
#### Evaluation
~~~
sh experiments/Dataset-Name/test.sh 
sh experiments/Dataset-Name/test.sh 
~~~

For cross-dataset testing run the relevant `change_vocab.py` script with the appropriate vocab size to generate the relevant vocabularies required for cross-dataset testing.


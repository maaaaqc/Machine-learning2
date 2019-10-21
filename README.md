## Machine Learning Group Project 1
* The packages will be installed when you call the shell scripts
* However, note that the name of the spacy "en" model might be different on linux/unix machines
* So a manual change on line 18 of Preprocessing.py might be required
* I will also include a Dockerfile that enables running it on docker
* However, no ports are specified, so to see the result excel file, manual change of Dockerfile and docker run command is required to share the file onto local machine

## Commands
* To reproduce the result:
```console
bash RunAll.sh
```
To see that the Bernoulli Model works:
```console
bash RunB.sh
```
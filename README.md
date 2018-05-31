Cette application est une API qui prend les données (pkl) sorti du syntagme extractor.

## Install les dépendances

```python
python -m pip install flask
python -m pip install flask-cors
```

**Note:** les dépendances dessus n'est que pour l'API, Si on a que besoin de lancer l'api, cela suiffira. Si on veut lancer le SyntagmExtractor, Il faut aussi installer ces dépendances.

## Lancer l'API

Pour windows, ouvrez un console, allez dans la répertoire du projet.

```shell
set FLASK_APP=api.py
flask run
```

Allez à http://127.0.0.1:5000      
Si vous voyez un Hello dans la navigateur, cela veut dire que  l'api est bien lancé.


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


## Utilisation de l'API

Avec cet API, on peut trouver les données qu'on veut par un requête HTTP, c'est à dire un URL avec quelques paramètres. Les donnes renvoyé sont sous format de JSON.     
Dans cette API, pour faciliter la compréhension, les mot sont standarlisé:

* <b>kernel1</b> : le noyau qui est présenté par un seul mot. exemple: numéro
* <b>kernel2</b> : le kernel1 enrichi. exemple: numéro de pervers
* <b>phrase</b> : c'est ce qu'on appelle syntagme. exemple: ce est sosh qui me donne un numéro de pervers
* <b>verbatime</b> : un verbatime complet qui contient la phrase

### structure des élément

#####kernel1
```json
{
    "kernel1": "contrôle",
    "score": 1.0
}
```

#####kernel2
```json
{
    "kernel2": "numéro de pervers",
    "phrases": [
        {
            "phrase": "ce est sosh qui me donne un numéro de pervers",
            "appearTimes": 1,
            "verbatimes": [
                {
                    "idx": 534,
                    "tags": [
                        "device"
                    ]
                }
            ],
            "cluster": "K0"
        }
    ]
}
```
Chaque kernel2 possède un liste de phrases.   

Un élément phrase a des attributs:

* appearTimes: nombre de fois que ce phrase apparaît.
* verbatims: un liste d'élément verbatime
* cluster: nom de cluster de cette phrase.

Un élément verbatime a des attributs:
* idx: index de ce verbatime dans le list de tous les verbatimes, ce numéro nous permet de accéder à ce verbatime.
* tags: un liste des tags pour ce verbatime.


### Exemple d'utilisation

* <b>tous les kernel1</b>: http://127.0.0.1:5000/kernel1s
* <b>tous les kernel2</b>: http://127.0.0.1:5000/kernel2s
* <b>filtrer les kernel2 qui contient le kernel1 facture</b>: http://127.0.0.1:5000/kernel2s?kernel1=facture
* <b>trouver les verbatimes avec index 20,30,40</b>: http://127.0.0.1:5000/docs?indexes=20,30,40
* 


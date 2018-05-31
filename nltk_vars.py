# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:13:53 2017

@author: KHZS7716
"""

telcoms = ["carte sim","réalité virtuelle","services client","service client","service assistance","service orange","service technique","service administratif","suivi consommation","suivi conso","mot de passe","adresse mail","orange france","airbox","rendez vous","orange et moi"]

monograms = [('m','me'),('d','de'),('c','cela'),('j','je'),('l','le'),('n','ne'),('s','se'),('t','te')]

singular = ['abcès','accès','abus','albatros','anchois','anglais','autobus','avis','brebis','carquois','cas','chas','colis','concours','corps','cours','cyprès','décès','devis','discours','dos','embarras','engrais','entrelacs','excès','fois','fonds','gâchis','gars','glas','guet-apens','héros','intrus','jars','jus','kermès','lacis','legs','lilas','marais','mars','matelas','mépris','mets','mois','mors','obus','os','palais','paradis','parcours','pardessus','pays','plusieurs','poids','pois','pouls','printemps','processus','progrès','puits','pus','rabais','radis','recors','recours','refus','relais','remords','remous','rhinocéros','repas','rubis','sas','secours','souris','succès','talus','tapis','taudis','temps','tiers','univers','velours','verglas','vernis','virus','accordailles','affres','aguets','alentours','ambages','annales','appointements','archives','armoiries','arrérages','arrhes','calendes','cliques','complies','condoléances','confins','dépens','ébats','entrailles','épousailles','errements','fiançailles','frais','funérailles','gens','honoraires','matines','mœurs','obsèques','pénates','pierreries','préparatifs','relevailles','rillettes','sévices','ténèbres','thermes','us','vêpres','victuailles','dès','envers','sous','vers','ailleurs','alors','après','certes','dedans','dehors','désormais','dessous','dessus','longtemps','moins','néanmoins','parfois','plus','puis','quelquefois','toutefois','toujours','très','volontiers','anglais','confus','frais','inclus','pers','des','plusieurs','vous','pas','sens','fils','soucis','français','nicolas']

stoplemmas = {"@card@","_file_","_@_","_url_","_ _","bonjour"}

my_stopwords = {"abord","afin","ah","ahlala","ai","aie","aient","aies","ailleurs","ainsi","ait","alors","Ap.","Apr.","après","as","assez","au","aucun","aucune","aujourd","auparavant","auprès","auquel","aura","aurai","auraient","aurais","aurait","auras","aurez","auriez","aurions","aurons","auront","aussi","aussitôt","autant","autour","autre","autres","autrui","aux","auxdites","auxdits","auxquelles","auxquels","avaient","avais","avait","avant","avec","avez","aviez","avions","avoir","avons","ayant","ayez","ayons","bah","banco","bé","beaucoup","ben","bien","bientôt","ça","car","ce","ceci","cela","celà","celle","celles","celui","cent","cents","cependant","certain","certaine","certaines","certains","ces","cet","cette","ceux","cf.","cg","cgr","chacun","chacune","chaque","chers","chez","chose","chut","ci","cinq","cinquante","cinquante-cinq","cinquante-deux","cinquante-et-un","cinquante-huit","cinquante-neuf","cinquante-quatre","cinquante-sept","cinquante-six","cinquante-trois","cl","cm","cm²","combien","comme","contre","crac","dans","davantage","de","d","dedans","dehors","déjà","delà","depuis","derrière","des","dès","desdites","desdits","desquelles","desquels","dessous","dessus","deux","devant","devers","dg","dix","dix-huit","dix-neuf","dix-sept","dl","dm","donc","dont","douze","du","dudit","duquel","durant","eh","elle","elles","en","encore","enfin","ensuite","entre","envers","es","ès","est","et","étaient","étais","était","étant","été","étée","étées","étés","êtes","étiez","étions","être","eu","eue","eues","euh","eûmes","eurent","eus","eusse","eussent","eusses","eussiez","eussions","eut","eût","eûtes","eux","fait","fi","flac","fois","fors","fûmes","furent","fus","fusse","fussent","fusses","fussiez","fussions","fut","fût","fûtes","GHz","gr","guère","ha","han","hé","hein","hélas","hem","heu","hg","hier","hl","hm","hm³","holà","hop","hormis","hors","hui","huit","hum","ici","il","ils","jamais","je","j","jusqu","jusque","kg","km","km²","la","là","laquelle","le","l","lequel","les","lès","lesquelles","lesquels","leur","leurs","lez","loin","longtemps","lors","lorsqu","lorsque","lui","m²","m³","ma","madame","maint","mainte","maintes","maints","mais","malgré","me","même","mêmes","mes","mg","mgr","MHz","mien","mienne","miennes","miens","mieux","mil","mille","ml","mm","mm²","moi","moins","mon","monsieur","moyennant","mt","ne","n","néanmoins","neuf","ni","nº","non","nonante","nonobstant","nos","notre","nôtre","nôtres","nous","np","nul","nulle","octante","oh","on","ont","onze","or","ou","où","ouais","outre","par","parce","parfois","parmi","pas","pendant","peu","plus","plusieurs","plutôt","pour","pourtant","pourvu","près","puisqu","puisque","qu","quand","quant","quarante","quarante-cinq","quarante-deux","quarante-et-un","quarante-huit","quarante-neuf","quarante-quatre","quarante-sept","quarante-six","quarante-trois","quatorze","quatre","quatre-vingt","quatre-vingt-cinq","quatre-vingt-deux","quatre-vingt-dix","quatre-vingt-dix-huit","quatre-vingt-dix-neuf","quatre-vingt-dix-sept","quatre-vingt-douze","quatre-vingt-huit","quatre-vingt-neuf","quatre-vingt-onze","quatre-vingt-quatorze","quatre-vingt-quatre","quatre-vingt-quinze","quatre-vingt-seize","quatre-vingt-sept","quatre-vingt-six","quatre-vingt-treize","quatre-vingt-trois","quatre-vingt-un","quatre-vingt-une","quatre-vingts","que","quel","quelle","quelles","quelqu","quelque","quelquefois","quelques","quels","qui","quiconque","quinze","quoi","quoiqu","quoique","revoici","revoilà","rien","sa","sans","sauf","se","seize","selon","sept","septante","sera","serai","seraient","serais","serait","seras","serez","seriez","serions","serons","seront","ses","seulement","si","sien","sienne","siennes","siens","sinon","sitôt","six","soi","soient","sois","soit","soixante","soixante-cinq","soixante-deux","soixante-dix","soixante-dix-huit","soixante-dix-neuf","soixante-dix-sept","soixante-douze","soixante-et-onze","soixante-et-un","soixante-et-une","soixante-huit","soixante-neuf","soixante-quatorze","soixante-quatre","soixante-quinze","soixante-seize","soixante-sept","soixante-six","soixante-treize","soixante-trois","sommes","son","sont","sous","souvent","soyez","soyons","suis","suite","sur","surtout","sus","ta","tandis","tant","tard","te","tel","telle","tellement","telles","tels","tes","tien","tion","toi","ton","toujours","tous","tout","toute","toutefois","toutes","treize","trente","trente-cinq","trente-deux","trente-et-un","trente-huit","trente-neuf","trente-quatre","trente-sept","trente-six","trente-trois","très","trois","trop","tu","un","une","unes","uns","USD","va","vers","via","vingt","vingt-cinq","vingt-deux","vingt-huit","vingt-neuf","vingt-quatre","vingt-sept","vingt-six","vingt-trois","vis-à-vis","voici","voilà","vos","votre","vôtre","vôtres","vous","ya","zéro","aah","aïe","air","ais","al","an","ans","at","axe","axé","ba","bal","ban","bas","bat","bb","bbr","bec","bel","bi","big","bis","bit","bla","bo","bol","bon","box","boy","btp","bue","bus","but","buy","cas","cd","cec","cf","cfr","cig","cis","clé","cm2","cn","co","coi","col","com","cos","cou","cpr","cru","crû","cse","cuq","cv","da","dam","dan","day","dei","del","dés","die","dis","dit","dit-il","dit-elle","dnc","do","doc","dom","don","dos","dpu","dru","dû","duc","due","dun","duo","dur","dus","ej","el","els","elu","élu","ème","eo","epi","épi","er","ère","etc","eté","eve","ex","fda","fe","fée","fer","feu","ff","fie","fil","fin","fir","fit","for","fr","fr3","gid","gît","go","gré","h00","h01","h1","h30","hic","hue","idd","ier","ii","iii","imd","in","ip","ir","ira","ire","is","its","iv","ive","ivv","jc","jp","k","ka","km3","kms","ko","kpn","l9","lac","lai","lan","lao","las","léo","lie","lié","lis","lit","lol","lot","low","lpm","ls","lu","luc","m2","mac","mae","mai","mal","man","mat","may","mds","men","mer","met","mi","mis","mme","mop","mr","mun","mus","my","n°1","n°5","n1","nai","nb","né","née","nés","net","nez","nh","nie","nié","no","not","npe","o","ô","ocm","ode","of","oie","one","ors","os","osa","ose","osé","ost","ôte","ôté","oui","our","pal","pan","pch","pe","pen","per","phu","pi","pie","pig","pim","pin","pis","pli","plm","po","pol","pop","pot","pou","ppf","pr","pre","pré","pro","pu","ras","rat","raz","rca","rda","rds","re","ré","res","ri","rif","rit","ro","rox","rps","sam","sd","sed","sel","sic","sid","sou","st","ste","su","sub","sui","sul","sûr","sut","sût","tac","tas","the","ths","ti","tic","tir","toe","tom","too","tôt","tre","tri","tuc","tue","tué","tus","tv","tvm","uck","uf","us","use","usé","uss","ux","val","van","vas","vau","ve","vè","ver","vie","vif","vil","vin","vis","vit","vol","von","vox","vtt","vu","vue","vus","vvr","way","wc","we","www","x","xv","xvè","xvi","xxe","xxè","xxi","you","zac","zag","zer","zig","ÿ","puis","co","rt","https","nhttps","http","amp","ok","bonne journée","merci"}

emotions=["Hâte", "hate", "envie", "kiffe", "kiffé", "kif", "kife", "impatience", "impatient", "impatiente", "omg", "cool", "aime", "adore", "deter", "j'kiffe", "jkiff", "j'kiffais", "jkiffais", "jkiffé", "j'kiffé", "jkiffai", "jkifferai", "j'kifferai", "jkifferais", "j'kifferais", "non", "noon", "nooon", "noooon", "noooooon", "bails", "bail", "accro", "addict", "accros", "addictif", "xd", "bon moment", "en paix", "préférée", "préféré", "préférer", "la meilleure", "la mieux", "la plus", "de fou", "fav", "favori", "favorite", "excellent", "ma vie", "détente", "détendre", "détendu", "détendue", "pleuré", "pleurer", "pleure", "larmes", "larme", "idéal", "idéale", "positive", "plasir", "ptikif", "magnifique", "génial", "géniale", "meilleure", "nul", "pourrie", "pas terrible", "moyen", "bof", "trop lent", "mauvais", "pas terrible", "naze", "envie", "wtf", "what the fuck", "dmieux", "boude", "boudé", "faché", "bouder", "facher", "véner", "enerver", "enervé", "NRV", "flemme", "flemmard", "flemmarde", "procrastination", "OKLM", "je devais", "j'aurai du", "j'aurais du", "jaurai du", "jaurais du", "pleure", "frissons", "frisson", "chair de poule", "jchiale", "je chiale", "seum", "dégouté", "dégoutée", "dégouter", "decu", "décue", "fier", "fierté", "fiere"]

tempos=["hier", "aujourd'hui", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche", "semaine", "mois", "matin", "soir", "midi", "aprem", "après-midi", "matinée", "soirée", "nuit", "en train", "ajd", "demain", "après-demain", "journée", "en même temps", "pendant ce temps", "pdt ce temps", "pdt que", "pendant que", "en mm tps", "en mm temps", "en attendant", "du mat", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h", "20h", "21h", "22h", "23h", "24h", "minuit", "midi", "lève", "couche", "H24", "sieste", "lever", "coucher", "levant", "couchant"]

blockbusters=["Westworld","WestworldHBO","stranger things","strangerthings","sense 8","sense8","renewsene8","#bringbacksense8","house of cards","houseofcards","Orange is the new black","orangeisthenewblack","walking dead","thewalkingdead","walkingdead","narcos","bureaux des légendes","daredevil","Luke cage","lukecage","thecrownnetflix","young pope","theyoungpope","theyoungpopefr","better call Saul","bettercallsaul","Big Bang theory","homeland","vikings","game of thrones","gameofthrones","#GoT","American horror story","AHSroanoke","the night of","thenightof","the get down","thegetdown","downtown abbey","downtownabbey","peaky blinders","peakyblinders","the flash","theflash","man in the high castle","man in the high castle","Fargo","13 reasons why","#13RW","Iron Fist","ironfirst","riverdale","Sherlock","orphelins de baudelaire","bettercallsaul","last man on earth","lastmanonearth","empire fox","empirefox","le secret d’elise","lesecretdelise","vengeance aux yeux clairs","LVAYC","person of interest","person of interest","Alice nevers","Alicenevers","flic tout simplement","flictoutsimplement","master of none","masterofnone","supergirl","versailles","exorcist","broadchurch","rick and morty","rick et morty","rickandmorty","rick&morty","the Americans","shameless","prison break","leftovers","prisonbreak","theAmericans","snatch"]

personnages=["Barry allen","Jon snow","saitama","frank underwood","clair underwood","claire underwood","frankunderwood","ClaireUnderwood","elliotalderson","ragnar"]

animes=["Lastman","onepunchman","dbz","Naruto","l’attaque des titans","full metal alchimist","korra","legend of korra","Gundam"]

films=["La La Land","LaLaLand","Logan","Fast & Furious","Fast and Furious","FastFurious","Resident Evil","ResidentEvil","batman vs superman","Fences","Trainspotting","Il a déjà tes yeux","À bras ouverts","Alibi.com","Django","Raid dingue","La Vallée des loups","Un profil pour deux","XXX Reactivated","Cinquante Nuances de Grey","Cinquante nuances plus sombres","Fifty Shades of Grey","FiftyShadesofGrey","star wars","starwars","zootopie","deadpool","premier contact","x-men","xmen","train pour busan","tarzan","livre de la jungle","ave cesar","8 salopards","the revenant"]
 
musiques=["Spotify","deezer","tidal","Apple Music","google play music","qobuz","soundcloud"]

connexions=["wifi","fibre","4g","3g","pas de réseau","plus de réseau","bande passante","connexion"]

lieux=["appart'","appartement","maison","lit","canap","canapé","fauteuil","chaise","table","cuisine","mon pieu","au pieu","matelas","lit","coussin","couette","toilette","toilettes","WC","chiotte","chiottes","cuisine","en cuisinant","chambre","terrasse","balcon","garage","grenier","salle de bain","sdb","sofa","salon","hall","table","chaises","voiture","taff","covoit","covoiturage","blablacar"]

mobilites=["metro","rer","tgv","train","bus","voiture","road trip","sur la route","chez vous","lycée","cours","avion","siège","fac","université","taff","taf","voyage","covoiturage","au boulot","au bureau","chez moi","chez eux","chez lui","chez elle","coloc'","coloc","colocation","fenetre","velux","baievitrée","baie vitrée","reflet","reflets","reflete","refleter","à coter","a coté","a ma gauche","a ma droite","aeroport","devant moi","derriere moi","gare","la salle","taff","taf","voyage","en réserve","en chemin","trottoir","banc","parc"]

avec_qui=["famille","papa","père","mere","maman","frère","soeur","soeurs","frères","copine","copain","mon mec","ma meuf","mon chérie","ma chérie","ma princesse","mon homme","pote","potes","poto","potos","amies","ami","amis","amie","cousin","cousins","cousines","cousine","oncle","tante","mamie","papi","colocs","coloc'","coloc","bébé","fils","fille","filles","enfants","gamins","les petits","les ptis","les gosses","chat","chien","hamster","lapin","les gars","les meufs","sœur","chaton","seul","tout seul","solo","mon frr","mes frr","bae","besta","bff","family","bro","brother","sister","sista","ma fille","mon fils","solo","frero","soeurette","reum","daron","darons","gamin"]

moments=["le bac","travail","boulot","sport","cours","je bosse","révise","reviser","révise mon code","revise le code","reviser le code","reviser mon code","passer le code","passer le permis","taff","école","récré","récréation","bureau"]

actions=["Posez","posey","posé","poser","tranquille","réconfort","après l'effort","dormir","lever","glander","chiller","chill","jmatte","j'matte","j'mattais","j'mattai","j""matté","jmattai","jmaté","je regarde","jregarde","j'regarde","jregardai","jregardé","j'regardais","j'regardai","j'regardais","mattes","regarder","pled","couverture","malade","angine","grippe","gastro","guele de bois","gueule de bois","cocooning","pépere","endormi","réveillé","concentrée","concentré","concentrer","fumer","fume","picole","boit","courir","marcher","je cours","running","footing","boxe","foot","tennis","rugby","yoga","méditation","pause","pédaler","pédale","vélo","ski","snow","snowboard","montagne","plage","serviette","stalk","enjailler","soirée","fête","teuf","danser","danse","je suis bien","chante","chanter","dansent","chantent","en chantant","en dansant","jmonte le son","son à donf","allonger","allongé","assis","assi","assise","couchée","posée"]

aliments=["vin","chocolat","thé","café","bière","bieres","repas","diner","gouter","pti dej","petit déjeuner","déjeuner","apéro","mcdo","quick","burger","burger","deliveroo","uber","foodora","manger","pancakes","crepes","tartine","nutella","kinder","milka","pates","gauffres","gateaux","brownie","starbucks","expresso","espresso","latte","soupe","burgers","pizzas","plats à emporter","plateaut télé","plateau tv"]

technos=["en HD","hd","en 4K","atmos","son à fond","5.1","dolby","1080p","son à donf","volume au max","son au max","HDMI","full hd","sonos","barre de son","bose","enceintes","écouteurs"]

devices=["ordi","tablette","ipad","iphone","xbox","box","freebox","sfrbox","sfrbox","bbox","b box","orange box","console","ordinateur","pc","mac","en replay","en live","redif","rediff","le player","le lecteur","mobile","smartphone","samsung","sony","nokia","ma surface","microsoft surface","tele","tv","mon tel","son tel","sur un site","dvd","blueray","blue ray","sur un site","écouteurs","casque","oreillette","samsung","mon macbook","mon mac","Chromecast","apple tv"]

douleurs=['erreur','erreurs','problème','problèmes','probleme','problemes','rpoblème','rpobleme','rpoblèmes','rpoblemes','porblème','porblèmes','porbleme','porblemes','défectueux','disfonctionnement','disfonctionnements','dysfonctionnement','dysfonctionnements','dépannage','dépannages','depannage','depannages','difficulté','difficultés','derangement','derangements','dérangement','dérangements','désagrément','désagréments','desagrement','desagrements','défaillance','défaillances','difficulté','difficultés','diffciulté','diffciultés','déboire','déboires','gloups','malheur','malheurs','restriction','restrictions','restriciton','restricitons','misère','misères','inquiétude','inquiétudes','panne','pannes','empechement','empêchement','empechements','empêchements','interruption','interruptions','vrille','vrilles',"ne marche pas","ne fonctionne pas",'ennui','ennuis']
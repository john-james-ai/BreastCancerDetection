Search.setIndex({"docnames": ["content/00_intro", "content/07_eda", "content/09_image_prep", "content/99_references"], "filenames": ["content/00_intro.md", "content/07_eda.ipynb", "content/09_image_prep.md", "content/99_references.md"], "titles": ["&lt;no title&gt;", "Exploratory Data Analysis (EDA) of the <strong>CBIS-DDSM</strong> Dataset", "Image Preprocessing", "References"], "terms": {"thi": [0, 1, 2], "small": [0, 1, 2], "sampl": [0, 2], "book": [0, 3], "give": [0, 2], "you": 0, "feel": 0, "how": [0, 1], "content": 0, "structur": 0, "It": [0, 1], "show": [0, 1], "off": 0, "few": [0, 1, 2], "major": [0, 1], "file": [0, 1, 2], "type": [0, 2, 3], "well": [0, 1, 2], "some": [0, 1, 2], "doe": [0, 1, 2], "go": 0, "depth": 0, "ani": [0, 1], "particular": [0, 1], "topic": 0, "check": [0, 1], "out": [0, 1], "jupyt": 0, "document": 0, "more": [0, 1, 2], "inform": [0, 1], "page": 0, "bundl": 0, "see": [0, 1], "exploratori": 0, "data": [0, 3], "analysi": 0, "eda": 0, "cbi": 0, "ddsm": 0, "dataset": 0, "imag": [0, 3], "preprocess": [0, 1], "refer": [0, 1], "In": [1, 2, 3], "section": [1, 2], "we": [1, 2], "conduct": [1, 2], "an": [1, 2], "prepar": 1, "prior": 1, "purpos": 1, "three": [1, 2], "fold": 1, "discov": 1, "relationship": 1, "among": [1, 2], "explor": [1, 2], "natur": [1, 2, 3], "between": [1, 2], "diagnost": 1, "properti": 1, "thei": [1, 2], "pertain": 1, "involv": 1, "follow": [1, 2], "contain": [1, 2], "patient": 1, "image_view": [1, 2], "i": [1, 2], "e": [1, 2, 3], "mammographi": [1, 2, 3], "roi": 1, "mask": 1, "crop": 1, "format": 1, "descript": 1, "1": [1, 2, 3], "patient_id": 1, "nomin": 1, "uniqu": [1, 2], "identifi": 1, "each": [1, 2], "2": [1, 2, 3], "breast_dens": 1, "discret": 1, "overal": 1, "volum": [1, 2, 3], "attenu": 1, "tissu": 1, "3": [1, 2, 3], "left_or_right_breast": 1, "which": [1, 2], "wa": [1, 2], "4": [1, 3], "dichotom": 1, "either": 1, "cranialcaud": 1, "mediolater": 1, "obliqu": 1, "5": [1, 2, 3], "abnormality_id": 1, "number": [1, 2], "6": [1, 2, 3], "abnormality_typ": [1, 2], "categori": [1, 2], "7": [1, 2, 3], "calc_typ": 1, "character": 1, "where": [1, 2], "applic": 1, "8": [1, 3], "calc_distribut": 1, "arrang": 1, "insid": 1, "rel": 1, "malign": 1, "9": [1, 3], "mass_shap": 1, "10": [1, 2, 3], "mass_margin": 1, "separ": 1, "from": [1, 2], "adjac": 1, "parenchyma": 1, "11": [1, 3], "12": [1, 3], "determin": [1, 2], "13": 1, "degre": [1, 2], "difficulti": 1, "14": 1, "fileset": 1, "indic": 1, "train": [1, 2], "test": 1, "set": [1, 3], "15": 1, "case_id": 1, "16": [1, 2], "whether": 1, "diagnos": 1, "As": [1, 2], "describ": [1, 2], "compound": 1, "morpholog": 1, "were": 1, "unari": 1, "dummi": 1, "encod": 1, "series_uid": 1, "seri": [1, 2], "filepath": 1, "path": [1, 2], "photometric_interpret": 1, "intend": 1, "samples_per_pixel": 1, "plane": 1, "row": 1, "column": 1, "aspect_ratio": 1, "continu": [1, 2], "vertic": [1, 2], "horizont": [1, 2], "bit_depth": 1, "store": [1, 2], "min_pixel_valu": 1, "minimum": 1, "actual": 1, "encount": 1, "largest_image_pixel": 1, "maximum": 1, "range_pixel_valu": 1, "differ": [1, 2], "largest": 1, "smallest": 1, "series_descript": 1, "full": [1, 2], "also": [1, 2], "includ": [1, 2], "far": 1, "better": [1, 2], "approxim": [1, 2], "answer": [1, 3], "often": [1, 2], "vagu": 1, "than": [1, 3], "exact": 1, "wrong": 1, "can": [1, 2], "alwai": 1, "made": 1, "precis": 1, "john": 1, "tukei": 1, "here": 1, "ll": [1, 2], "put": 1, "forward": 1, "motiv": 1, "discoveri": 1, "process": [1, 2], "what": 1, "ar": [1, 2], "To": 1, "relat": [1, 2], "certain": 1, "less": 1, "subtl": 1, "concern": 1, "do": 1, "have": [1, 2], "standard": [1, 2], "represent": 1, "term": [1, 2], "artifact": [1, 2], "mark": 1, "text": [1, 2], "extant": 1, "would": 1, "bright": [1, 2], "contrast": [1, 2], "primari": 1, "stage": 1, "preliminari": 1, "ha": [1, 2], "python": 1, "packag": [1, 2], "depend": [1, 2], "panda": 1, "tabular": 1, "numpi": 1, "numer": 1, "matplotlib": 1, "seaborn": 1, "visual": 1, "scipi": 1, "statist": [1, 2], "studioai": 1, "pd": 1, "stat": 1, "pickl": 1, "np": 1, "pyplot": 1, "plt": 1, "sn": 1, "object": [1, 2], "so": 1, "sklearn": 1, "svm": 1, "svc": 1, "linear_model": 1, "logisticregress": 1, "ensembl": 1, "randomforestclassifi": 1, "bcd": [1, 2], "analyz": 1, "caseexplor": 1, "dicomexplor": 1, "pipelinebuild": 1, "modelselector": 1, "option": 1, "displai": 1, "max_row": 1, "999": 1, "max_column": 1, "set_styl": 1, "whitegrid": 1, "set_palett": 1, "blues_r": 1, "case_fp": 1, "calc": 1, "df": 1, "get_calc_data": 1, "get_mass_data": 1, "compris": [1, 2], "analys": 1, "let": [1, 2], "s": [1, 2, 3], "sens": 1, "1566": 1, "3566": 1, "1872": 1, "1199": [1, 2], "673": 1, "1694": 1, "910": 1, "784": 1, "st": 1, "t": [1, 2], "pct_calc": 1, "round": 1, "100": [1, 2], "0": [1, 2, 3], "pct_mass": 1, "pct_calc_mal": 1, "pct_calc_bn": 1, "pct_mass_mal": 1, "pct_mass_bn": 1, "cases_per_pati": 1, "msg": [1, 2], "f": [1, 2, 3], "kei": 1, "observ": [1, 2], "n": [1, 2], "tthe": 1, "comport": 1, "tcia": 1, "twe": 1, "tof": 1, "ton": 1, "averag": [1, 2], "print": 1, "52": 1, "47": 1, "Of": 1, "64": 1, "05": [1, 2, 3], "35": 1, "95": 1, "53": 1, "72": 1, "46": 1, "28": 1, "On": 1, "take": 1, "look": 1, "1456": 1, "train_p_01885_right_mlo_1": 1, "p_01885": 1, "mlo": 1, "lucent_cent": 1, "segment": 1, "benign_without_callback": 1, "fals": [1, 2], "1378": 1, "train_p_01809_left_mlo_1": 1, "p_01809": 1, "pleomorph": 1, "true": 1, "1432": 1, "train_p_01858_left_mlo_1": 1, "p_01858": 1, "punctat": 1, "cluster": 1, "1428": 1, "train_p_01847_left_mlo_1": 1, "p_01847": 1, "linear": [1, 2], "1183": 1, "train_p_01488_left_mlo_1": 1, "p_01488": 1, "2863": 1, "train_p_01418_right_mlo_1": 1, "p_01418": 1, "irregular": [1, 3], "spicul": 1, "2315": 1, "train_p_00640_right_cc_1": 1, "p_00640": 1, "cc": 1, "ill_defin": 1, "2391": 1, "train_p_00739_left_cc_1": 1, "p_00739": 1, "lobul": 1, "circumscrib": [1, 3], "2661": 1, "train_p_01150_left_cc_1": 1, "p_01150": 1, "obscur": [1, 3], "2815": 1, "train_p_01356_left_cc_1": 1, "p_01356": 1, "our": [1, 2], "cover": 1, "radiologist": 1, "classifi": [1, 2], "us": [1, 2, 3], "level": [1, 2], "scale": [1, 2], "almost": 1, "entir": 1, "fatti": 1, "scatter": 1, "area": [1, 2], "fibroglandular": 1, "heterogen": 1, "dens": [1, 3], "extrem": 1, "note": 1, "correspond": 1, "b": [1, 3], "c": [1, 2], "d": [1, 2], "list": 1, "abov": [1, 2], "confus": 1, "notwithstand": 1, "ordin": 1, "chart": 1, "illustr": [1, 2], "within": [1, 2], "fig": 1, "ax": 1, "subplot": 1, "figsiz": 1, "plot": 1, "countplot": 1, "x": [1, 2], "titl": 1, "plot_count": 1, "balanc": 1, "respect": 1, "digit": [1, 3], "two": [1, 2], "cranial": 1, "caudal": 1, "taken": 1, "best": 1, "subarcolar": 1, "central": [1, 2], "medial": 1, "posteromedi": 1, "project": [1, 2], "its": [1, 2], "entireti": 1, "posterior": 1, "upper": 1, "outer": 1, "quadrant": 1, "proport": 1, "sequenc": 1, "assign": 1, "count": 1, "vast": [1, 2], "present": 1, "singl": [1, 2], "although": 1, "consider": 1, "common": 1, "mammogram": [1, 3], "especi": 1, "after": 1, "ag": 1, "50": 1, "calcium": 1, "deposit": 1, "typic": [1, 2], "up": 1, "macrocalcif": 1, "microcalcif": 1, "appear": [1, 2], "larg": [1, 2], "white": [1, 2], "dot": [1, 2], "dash": 1, "noncancer": 1, "requir": [1, 2], "further": [1, 2], "fine": 1, "speck": 1, "similar": [1, 2], "grain": 1, "salt": [1, 2], "usual": [1, 2], "pattern": 1, "earli": 1, "sign": 1, "particularli": 1, "women": 1, "reproduct": 1, "For": [1, 2], "25": 1, "affect": [1, 2], "diseas": 1, "lifetim": 1, "initi": 1, "new": [1, 2], "care": 1, "wide": [1, 2], "rang": [1, 2], "caus": [1, 2], "physiolog": 1, "adenosi": 1, "aggress": [1, 3], "shown": [1, 2], "below": 1, "measur": 1, "difficult": 1, "obviou": 1, "17": 1, "A": [1, 2, 3], "plural": 1, "moder": 1, "slightli": 1, "nearli": 1, "3rd": 1, "consid": [1, 2], "base": 1, "upon": 1, "thorough": 1, "evalu": [1, 2], "mammograph": [1, 3], "six": 1, "definit": 1, "mean": 1, "find": 1, "unclear": 1, "need": [1, 2], "score": 1, "neg": 1, "normal": 1, "No": 1, "asymmetri": 1, "other": 1, "been": [1, 2], "found": [1, 2], "while": [1, 2], "mai": [1, 2, 3], "detect": [1, 3], "most": [1, 2], "like": [1, 2], "suspect": 1, "four": 1, "subcategori": 1, "4a": 1, "4b": 1, "4c": 1, "chanc": 1, "higher": [1, 2], "being": 1, "previous": 1, "biopsi": 1, "factor": 1, "differenti": 1, "There": 1, "over": [1, 2], "40": 1, "main": 1, "amorph": 1, "indistinct": 1, "without": 1, "clearli": 1, "defin": 1, "hazi": 1, "coars": 1, "conspicu": [1, 3], "larger": [1, 2], "mm": 1, "dystroph": 1, "lava": 1, "develop": 1, "year": 1, "treatment": 1, "about": 1, "30": 1, "eggshel": 1, "veri": 1, "thin": 1, "branch": 1, "curvilinear": 1, "rod": 1, "form": [1, 2], "occassion": 1, "lucent": 1, "center": 1, "oval": 1, "fat": 1, "necrosi": 1, "calcifi": 1, "debri": 1, "duct": 1, "milk": 1, "sediment": 1, "macro": 1, "microcyst": 1, "vari": 1, "skin": 1, "vascular": 1, "parallel": 1, "track": 1, "blood": 1, "vessel": 1, "y": [1, 2], "order_by_count": 1, "account": 1, "half": 1, "75": 1, "repres": [1, 2], "five": 1, "diffus": [1, 2], "throughout": 1, "whole": 1, "region": [1, 2], "expect": 1, "ductal": 1, "group": 1, "least": 1, "lobe": 1, "80": 1, "calfic": 1, "lexicon": 1, "howev": [1, 2], "addit": [1, 2], "symmetri": 1, "architectur": 1, "low": [1, 2], "undetermin": 1, "likelihood": 1, "microlobul": 1, "carcinoma": 1, "ill": [1, 2], "call": [1, 2], "gener": [1, 2], "make": 1, "70": 1, "distinguish": 1, "outcom": [1, 3], "callback": 1, "latter": 1, "should": 1, "monitor": 1, "investig": 1, "collaps": 1, "sever": 1, "fall": 1, "one": [1, 2], "similarli": 1, "20": 1, "yet": [1, 2], "class": 1, "next": 1, "inter": 1, "former": 1, "explanatori": 1, "independ": 1, "as_df": 1, "categorize_ordin": 1, "color": [1, 2], "add": 1, "bar": 1, "stack": 1, "theme": 1, "axes_styl": 1, "grid": 1, "linestyl": 1, "label": 1, "layout": 1, "engin": 1, "tight": 1, "rather": 1, "prop": 1, "groupbi": [1, 2], "value_count": 1, "to_fram": 1, "reset_index": 1, "sort_valu": 1, "risk": 1, "don": 1, "reveal": 1, "strong": 1, "support": [1, 2], "infer": 1, "kt": 1, "kendallstau": 1, "name": [1, 2], "kendal": 1, "\u03c4": 1, "011138411035362646": 1, "pvalu": 1, "5382894688881223": 1, "alpha": 1, "strength": 1, "weak": 1, "tau": 1, "non": [1, 2], "signific": 1, "effect": [1, 2], "phi_": 1, "01": 1, "p": [1, 3], "54": 1, "2022": [1, 3], "studi": 1, "publish": 1, "suggest": 1, "preval": 1, "bodi": 1, "If": 1, "greater": [1, 2], "evid": 1, "cv": 1, "cramersv": 1, "cramer": 1, "v": 1, "028861432861833916": 1, "08480010265447133": 1, "neglig": 1, "dof": 1, "x2alpha": 1, "x2": 1, "970414906184831": 1, "x2dof": 1, "chi": 1, "squar": [1, 2], "97": 1, "08": 1, "phi": 1, "03": [1, 3], "rsna": [1, 3], "journal": 1, "high": 1, "craniocaud": 1, "59": 1, "41": 1, "both": 1, "same": [1, 2], "0014163311721585292": 1, "9325971883801198": 1, "007153374565586881": 1, "007": 1, "93": 1, "002": 1, "Is": 1, "57": 1, "43": 1, "These": [1, 2], "10437040964253724": 1, "5880686528820935e": 1, "38": 1, "84508847031938": 1, "85": 1, "compar": 1, "vs": 1, "di": 1, "agreement": 1, "report": [1, 3], "seven": 1, "incomplet": 1, "comparison": 1, "na": 1, "routin": 1, "essenti": [1, 2], "short": 1, "interv": 1, "month": 1, "known": [1, 2], "proven": 1, "concat": 1, "axi": [1, 2], "177": [1, 3], "00": 1, "642": 1, "375": 1, "79": 1, "102": [1, 2], "21": 1, "902": 1, "55": 1, "731": 1, "45": 1, "02": [1, 3], "560": 1, "98": [1, 2], "5994799138998625": 1, "4696313517612682e": 1, "244": 1, "inde": 1, "60": 1, "had": 1, "onli": 1, "all": [1, 2], "ultim": [1, 2], "just": 1, "isn": 1, "clear": 1, "examin": 1, "vi": 1, "seem": 1, "78": 1, "96": 1, "232": 1, "207": 1, "627": 1, "65": 1, "337": 1, "559": 1, "316": 1, "36": 1, "613": 1, "501": 1, "again": 1, "draw": 1, "003196827770471352": 1, "8618112089236021": 1, "003": [1, 3], "86": 1, "accord": 1, "literatur": [1, 2], "highest": [1, 2], "df_calc": 1, "5363368552127653": 1, "078377585363777e": 1, "87": 1, "538": 1, "4943200698192": 1, "42": 1, "539": 1, "69": [1, 3], "top": 1, "get_most_malignant_calc": 1, "barplot": 1, "32634130163729136": 1, "693260459198279e": 1, "39": 1, "199": 1, "36546372889003": 1, "198": 1, "56": 1, "33": 1, "df_mass": 1, "get_most_malignant_mass": 1, "510182454781321": 1, "297593104510473e": 1, "81": 1, "440": 1, "9247163603807": 1, "19": 1, "92": 1, "51": 1, "enabl": 1, "5894681913733985": 1, "1871720584088994e": 1, "113": 1, "588": 1, "6188361978973": 1, "18": 1, "62": 1, "That": 1, "conclud": 1, "impli": 1, "exercis": [1, 2], "start": 1, "Then": 1, "avoid": 1, "spuriou": 1, "across": 1, "plot_feature_associ": 1, "ignor": 1, "associationss": 1, "now": [1, 2], "said": 1, "df_prop": 1, "sort": 1, "hue": 1, "tend": 1, "behav": 1, "thu": 1, "plot_calc_feature_associ": 1, "_": 1, "summarize_morphology_by_featur": 1, "render": 1, "suspicion": 1, "those": 1, "intermedi": 1, "remain": [1, 2], "classif": 1, "5399745903685791": 1, "2183": 1, "2953161289365": 1, "168": 1, "signfic": 1, "therebi": 1, "reduc": [1, 2], "compare_morpholog": 1, "m1": 1, "m2": 1, "co": [1, 2], "occur": 1, "instanc": [1, 2], "exclus": 1, "45403619931007766": 1, "3087": 1, "2854813722943": 1, "336": 1, "cours": 1, "large_rodlik": 1, "regular": [1, 2], "32542688359496785": 1, "164653782848766e": 1, "82": 1, "792": 1, "9990923686997": 1, "strongli": 1, "793": 1, "anyth": 1, "primarili": [1, 2], "38789958530499524": 1, "1562872500601785e": 1, "216": 1, "1126": 1, "690069039047": 1, "32": 1, "1127": 1, "middl": 1, "rodlik": 1, "stand": 1, "specif": 1, "3552474787413507": 1, "624419427603362e": 1, "708": 1, "7435307901172": 1, "126": 1, "obser": 1, "709": 1, "plot_mass_feature_associ": 1, "asses": 1, "37": 1, "notabl": 1, "weakli": 1, "perhap": 1, "40037910751937816": 1, "3306037914870686e": 1, "225": 1, "1357": 1, "7700498809766": 1, "90": 1, "1358": 1, "distort": 1, "3693905740902926": 1, "3746514669523e": 1, "182": 1, "1155": 1, "7263860406229": 1, "1156": 1, "compon": 1, "summar": 1, "part": 1, "pair": 1, "plot_target_associ": 1, "depict": 1, "mani": 1, "updat": 1, "gather": 1, "physician": 1, "strongest": 1, "rate": 1, "exceed": 1, "elucid": 1, "beyond": [1, 2], "deriv": [1, 2], "therefor": [1, 2], "impact": 1, "estim": [1, 2], "explain": 1, "establish": 1, "given": [1, 2], "logist": 1, "regress": 1, "vector": 1, "machin": 1, "random": [1, 2], "forest": 1, "cross": 1, "valid": 1, "algorithm": 1, "built": 1, "pb": 1, "set_job": 1, "set_standard_scal": 1, "set_scor": 1, "accuraci": 1, "params_lr": 1, "clf__penalti": 1, "l1": 1, "l2": 1, "clf__c": 1, "clf__solver": 1, "liblinear": 1, "clf": 1, "random_st": 1, "set_classifi": 1, "param": [1, 2], "build_gridsearch_cv": 1, "lr": 1, "params_svc": 1, "clf__kernel": 1, "param_rang": 1, "params_rf": 1, "clf__criterion": 1, "gini": 1, "entropi": 1, "clf__min_samples_leaf": 1, "clf__max_depth": 1, "clf__min_samples_split": 1, "rf": 1, "x_train": 1, "y_train": 1, "x_test": 1, "y_test": 1, "get_calc_model_data": 1, "best_calc_model_fp": 1, "os": [1, 2], "abspath": [1, 2], "best_calc_pipelin": 1, "pkl": 1, "selector": 1, "calc_m": 1, "add_pipelin": 1, "run": [1, 2], "forc": 1, "force_model_fit": 1, "y_pred": 1, "y_true": 1, "load": 1, "recal": 1, "f1": 1, "73": 1, "83": 1, "avg": 1, "66": 1, "67": 1, "weight": [1, 2], "76": [1, 3], "71": 1, "outperform": 1, "achiev": 1, "posit": [1, 2], "abil": 1, "58": 1, "harmon": 1, "coeffici": 1, "task": [1, 2], "provid": 1, "belong": 1, "wherea": 1, "ncalcif": 1, "nfeatur": 1, "plot_feature_import": 1, "greatest": 1, "align": 1, "current": [1, 2], "understood": 1, "get_mass_model_data": 1, "best_mass_model_fp": 1, "best_mass_pipelin": 1, "mass_m": 1, "77": 1, "84": 1, "perform": [1, 2], "nmass": 1, "unlik": 1, "against": 1, "comput": [1, 2, 3], "increas": [1, 2], "puriti": 1, "leav": 1, "tree": 1, "pure": 1, "point": [1, 2], "line": [1, 2], "length": 1, "thick": 1, "radiat": 1, "sharp": [1, 2], "demarc": 1, "lesion": 1, "surround": 1, "neither": 1, "nor": 1, "hidden": 1, "fibro": 1, "glandular": 1, "henc": [1, 2], "cannot": 1, "fulli": 1, "practic": [1, 2, 3], "commonli": [1, 2, 3], "when": [1, 2], "portion": 1, "lower": [1, 2], "avail": 1, "ds": 1, "dicom_fp": 1, "info": 1, "datatyp": 1, "complet": 1, "null": 1, "duplic": 1, "uid": 1, "3565": 1, "3100": 1, "465": 1, "331545": 1, "430492": 1, "606875": 1, "3564": 1, "3633": 1, "int32": 1, "14260": 1, "int64": 1, "349": 1, "3216": 1, "28520": 1, "425": 1, "3140": 1, "2591": 1, "974": 1, "float64": 1, "2595": 1, "970": 1, "3624": 1, "max_pixel_valu": 1, "233": 1, "3332": 1, "07": 1, "mean_pixel_valu": 1, "3031": 1, "534": 1, "median_pixel_valu": 1, "1086": 1, "2479": 1, "std_pixel_valu": 1, "278070": 1, "1999": 1, "44": 1, "228160": 1, "3561": 1, "3563": 1, "3796": 1, "3684": 1, "3558": 1, "22": 1, "234313": 1, "23": 1, "3521": 1, "253086": 1, "24": [1, 3], "3555": 1, "244027": 1, "3559": 1, "26": 1, "3562": 1, "240560": 1, "27": 1, "3560": 1, "220326": 1, "29": [1, 3], "3544": 1, "245884": 1, "3545": 1, "249740": 1, "31": 1, "322810": 1, "bool": 1, "categor": [1, 2], "referenc": 1, "focu": [1, 2], "monochrome2": 1, "evenli": 1, "df_full": 1, "loc": 1, "nrow": 1, "ncol": 1, "histogram": 1, "suptitl": 1, "nimag": 1, "tight_layout": 1, "transform": [1, 2], "interest": 1, "df_full_desc": 1, "datafram": 1, "std": 1, "min": 1, "max": 1, "565": 1, "926": 1, "946": 1, "968": 1, "535": 1, "zero": [1, 2], "inspect": 1, "nois": [1, 2], "annot": 1, "interclass": 1, "dissimilar": 1, "intraclass": 1, "organ": [1, 2], "variou": 1, "plot_imag": 1, "thirteen": 1, "remov": [1, 2], "condit": 1, "lambda": 1, "isin": 1, "optim": 2, "deep": 2, "learn": 2, "step": 2, "pector": 2, "muscl": 2, "enhanc": 2, "medic": 2, "experiment": 2, "experi": 2, "appli": 2, "befor": 2, "model": 2, "qualiti": 2, "extract": 2, "multivari": 2, "stratifi": 2, "first": 2, "import": 2, "modul": 2, "jbook": 2, "getcwd": 2, "chdir": 2, "join": 2, "config": 2, "bcdcontain": 2, "etl": 2, "loader": 2, "denoiseexperi": 2, "bilateralfilt": 2, "medianfilt": 2, "wire": 2, "init_resourc": 2, "dal": 2, "repo": 2, "io": 2, "paramet": 2, "setup_complet": 2, "denoise_complet": 2, "batchsiz": 2, "reset": 2, "uow": 2, "stdinnotimplementederror": 2, "traceback": 2, "recent": 2, "last": 2, "cell": 2, "py": 2, "self": 2, "mode": 2, "99": 2, "restor": 2, "state": 2, "proce": 2, "input": 2, "101": 2, "image_repo": 2, "delete_by_mod": 2, "anaconda3": 2, "env": 2, "lib": 2, "python3": 2, "site": 2, "ipykernel": 2, "kernelbas": 2, "1201": 2, "raw_input": 2, "prompt": 2, "_allow_stdin": 2, "1200": 2, "frontend": 2, "request": 2, "rais": 2, "1202": 2, "return": 2, "_input_request": 2, "1203": 2, "str": 2, "1204": 2, "_parent_id": 2, "shell": 2, "1205": 2, "get_par": 2, "1206": 2, "password": 2, "1207": 2, "abnorm": [2, 3], "view": 2, "bi": [2, 3], "rad": [2, 3], "assess": 2, "cancer": [2, 3], "diagnosi": [2, 3], "frac": 2, "variat": 2, "produc": 2, "dure": 2, "captur": 2, "fluctuat": 2, "pepper": 2, "speckl": 2, "poisson": 2, "spike": 2, "impuls": 2, "flat": 2, "tail": 2, "distribut": 2, "black": 2, "mainli": 2, "radar": 2, "wherebi": 2, "signal": 2, "final": 2, "shot": 2, "due": 2, "characterist": 2, "devic": 2, "photon": 2, "dose": 2, "rai": 2, "mathemat": 2, "noisi": 2, "unknown": 2, "clean": 2, "awgn": 2, "deviat": 2, "sigma_n": 2, "sinc": 2, "solut": 2, "pose": 2, "replet": 2, "techniqu": 2, "hat": 2, "survei": 2, "scope": 2, "effort": 2, "introduc": 2, "inher": 2, "challeng": 2, "ensur": 2, "smooth": 2, "protect": 2, "edg": 2, "blur": 2, "preserv": 2, "textur": 2, "effici": 2, "time": 2, "complex": 2, "roughli": 2, "spatial": 2, "domain": 2, "replac": 2, "pixel": 2, "valu": 2, "neighbor": 2, "The": [2, 3], "origin": 2, "simpl": 2, "computation": 2, "intuit": 2, "local": 2, "simpli": 2, "itself": 2, "specifi": 2, "shape": 2, "size": 2, "neighborhood": 2, "must": 2, "odd": 2, "integ": 2, "direct": 2, "work": 2, "convolv": 2, "intens": 2, "output": 2, "easi": 2, "implement": 2, "drawback": 2, "outlier": 2, "significantli": 2, "problemat": 2, "convolut": 2, "oper": 2, "By": 2, "isotrop": 2, "circularli": 2, "symmetr": 2, "g": [2, 3], "sigma": 2, "pi": 2, "distanc": 2, "assum": 2, "amount": 2, "spread": 2, "function": 2, "lesser": 2, "collect": 2, "approach": 2, "review": 2, "constant": 2, "integr": 2, "uniti": 2, "everi": 2, "grei": 2, "substanti": 2, "amplitud": 2, "respons": 2, "invari": 2, "get": 2, "wider": 2, "varianc": 2, "sum": 2, "constitut": 2, "wai": 2, "concaten": 2, "creat": 2, "analog": 2, "waterfal": 2, "span": 2, "height": 2, "total": 2, "dimension": 2, "product": 2, "g_": 2, "2d": 2, "2_1": 2, "2_2": 2, "1d": 2, "sigma_1": 2, "otim": 2, "sigma_2": 2, "element": 2, "move": 2, "One": 2, "popular": 2, "l": [2, 3], "displaystyl": 2, "sum_": 2, "infti": 2, "i_n": 2, "denot": 2, "modifi": 2, "bessel": 2, "sine": 2, "counterpart": 2, "equat": 2, "space": 2, "revisit": 2, "theori": 2, "infinit": 2, "enough": 2, "truncat": 2, "m": 2, "chosen": 2, "somewher": 2, "altern": 2, "frequenc": 2, "leverag": 2, "close": 2, "express": 2, "fourier": 2, "theta": 2, "reduct": 2, "breast": 3, "system": 3, "url": 3, "http": 3, "www": 3, "acr": 3, "org": 3, "clinic": 3, "resourc": 3, "visit": 3, "2023": 3, "09": 3, "shelli": 3, "lill\u00e9": 3, "wendi": 3, "marshal": 3, "valeri": 3, "andolina": 3, "guid": 3, "wolter": 3, "kluwer": 3, "philadelphia": 3, "fourth": 3, "edit": 3, "2019": 3, "isbn": 3, "978": 3, "4963": 3, "5202": 3, "oclc": 3, "1021062474": 3, "ask": 3, "question": 3, "nci": 3, "februari": 3, "2018": 3, "archiv": 3, "locat": 3, "nciglob": 3, "ncienterpris": 3, "gov": 3, "chang": 3, "yara": 3, "abdou": 3, "medhavi": 3, "gupta": 3, "mariko": 3, "asaoka": 3, "kristoph": 3, "attwood": 3, "opyrch": 3, "mateusz": 3, "shipra": 3, "gandhi": 3, "kazuaki": 3, "takab": 3, "left": 3, "side": 3, "associ": 3, "biologi": 3, "wors": 3, "right": 3, "scientif": 3, "13377": 3, "august": 3, "com": 3, "articl": 3, "s41598": 3, "022": 3, "16749": 3, "doi": 3, "1038": 3, "katrina": 3, "korhonen": 3, "emili": 3, "conant": 3, "eric": 3, "cohen": 3, "mari": 3, "synnestvedt": 3, "elizabeth": 3, "mcdonald": 3, "susan": 3, "weinstein": 3, "simultan": 3, "acquir": 3, "versu": 3, "tomosynthesi": 3, "radiolog": 3, "292": 3, "juli": 3, "pub": 3, "1148": 3, "radiol": 3, "2019182027": 3, "lawrenc": 3, "w": 3, "bassett": 3, "karen": 3, "conner": 3, "iv": 3, "ms": 3, "holland": 3, "frei": 3, "medicin": 3, "6th": 3, "bc": 3, "decker": 3, "2003": 3, "ncbi": 3, "nlm": 3, "nih": 3, "nbk12642": 3, "rebecca": 3, "sawyer": 3, "lee": 3, "francisco": 3, "gimenez": 3, "assaf": 3, "hoogi": 3, "kana": 3, "kawai": 3, "miyak": 3, "mia": 3, "gorovoi": 3, "daniel": 3, "rubin": 3, "curat": 3, "aid": 3, "research": 3, "170177": 3, "decemb": 3, "2017": 3, "sdata2017177": 3, "sdata": 3, "david": 3, "bragg": 3, "philip": 3, "hedvig": 3, "hricak": 3, "oncolog": 3, "saunder": 3, "2nd": 3, "ed": 3, "2002": 3, "7216": 3, "7494": 3, "erin": 3, "princ": 3, "heidi": 3, "r": 3, "umphrei": 3, "multipl": 3, "mass": 3, "oxford": 3, "univers": 3, "press": 3, "march": 3, "academ": 3, "oup": 3, "24629": 3, "chapter": 3, "187959266": 3, "1093": 3, "med": 3, "9780190270261": 3, "0019": 3, "187965422": 3, "0026": 3, "elezabi": 3, "187962148": 3, "0023": 3}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"exploratori": 1, "data": [1, 2], "analysi": 1, "eda": 1, "cbi": 1, "ddsm": 1, "dataset": 1, "The": 1, "case": 1, "dictionari": 1, "dicom": 1, "approach": 1, "guid": 1, "question": 1, "plan": 1, "univari": 1, "breast": 1, "densiti": 1, "left": 1, "right": 1, "side": 1, "imag": [1, 2], "view": 1, "abnorm": 1, "id": 1, "type": 1, "subtleti": 1, "bi": 1, "rad": 1, "assess": 1, "calcif": 1, "distribut": 1, "mass": 1, "shape": 1, "margin": 1, "patholog": 1, "cancer": 1, "summari": 1, "bivari": 1, "target": 1, "variabl": 1, "associ": 1, "diagnosi": 1, "featur": 1, "multivari": 1, "build": 1, "pipelin": 1, "get": 1, "model": 1, "execut": 1, "select": 1, "predict": 1, "result": 1, "import": 1, "metadata": 1, "photometr": 1, "interpret": 1, "sampl": 1, "per": 1, "pixel": 1, "bit": 1, "depth": 1, "height": 1, "width": 1, "size": 1, "aspect": 1, "ratio": 1, "valu": 1, "qualiti": 1, "benign": 1, "probabl": 1, "suspici": 1, "highli": 1, "preprocess": 2, "setup": 2, "initi": 2, "repositori": 2, "load": 2, "denois": 2, "problem": 2, "statement": 2, "method": 2, "meanfilt": 2, "guassianfilt": 2, "normal": 2, "cascad": 2, "properti": 2, "separ": 2, "discret": 2, "gaussian": 2, "kernel": 2, "gaussianfilt": 2, "mean": 2, "filter": 2, "median": 2, "bilater": 2, "refer": 3}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})
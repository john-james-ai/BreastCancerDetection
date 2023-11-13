Search.setIndex({"docnames": ["content/00_intro", "content/06_eda", "content/99_references"], "filenames": ["content/00_intro.md", "content/06_eda.ipynb", "content/99_references.md"], "titles": ["&lt;no title&gt;", "Exploratory Data Analysis (EDA) of the <strong>CBIS-DDSM</strong> Dataset", "References"], "terms": {"thi": [0, 1], "small": [0, 1], "sampl": 0, "book": [0, 2], "give": 0, "you": 0, "feel": 0, "how": [0, 1], "content": 0, "structur": 0, "It": [0, 1], "show": [0, 1], "off": 0, "few": [0, 1], "major": [0, 1], "file": [0, 1], "type": [0, 2], "well": [0, 1], "some": [0, 1], "doe": [0, 1], "go": 0, "depth": 0, "ani": [0, 1], "particular": [0, 1], "topic": 0, "check": [0, 1], "out": [0, 1], "jupyt": 0, "document": [0, 2], "more": [0, 1], "inform": [0, 1], "page": 0, "bundl": 0, "see": [0, 1], "exploratori": 0, "data": [0, 2], "analysi": 0, "eda": 0, "cbi": 0, "ddsm": 0, "dataset": 0, "refer": [0, 1], "In": [1, 2], "section": 1, "we": 1, "conduct": 1, "an": 1, "prepar": 1, "prior": 1, "purpos": 1, "four": 1, "fold": 1, "discov": 1, "relationship": 1, "among": 1, "explor": 1, "natur": [1, 2], "degre": 1, "which": 1, "relat": 1, "identifi": 1, "error": 1, "outlier": 1, "anomali": 1, "befor": 1, "stage": 1, "develop": 1, "appropri": 1, "pre": 1, "process": [1, 2], "involv": 1, "follow": 1, "contain": 1, "patient": 1, "image_view": 1, "properti": 1, "mammographi": [1, 2], "roi": 1, "mask": 1, "crop": 1, "format": 1, "descript": 1, "valid": 1, "1": [1, 2], "patient_id": 1, "nomin": 1, "uniqu": 1, "each": 1, "string": 1, "p_00000": 1, "2": [1, 2], "breast_dens": 1, "discret": 1, "overal": 1, "volum": [1, 2], "attenu": 1, "tissu": 1, "integ": 1, "rang": 1, "4": [1, 2], "3": [1, 2], "left_or_right_breast": 1, "wa": 1, "dichotom": 1, "either": 1, "cranialcaud": 1, "mediolater": 1, "obliqu": 1, "cc": 1, "mlo": 1, "5": [1, 2], "abnormality_id": 1, "number": 1, "6": [1, 2], "abnormality_typ": 1, "categori": 1, "7": [1, 2], "calc_typ": 1, "character": 1, "where": 1, "applic": 1, "appendix": 1, "8": [1, 2], "calc_distribut": 1, "arrang": 1, "insid": 1, "rel": 1, "malign": 1, "9": [1, 2], "0": [1, 2], "10": [1, 2], "determin": 1, "benign_without_callback": 1, "11": [1, 2], "diagnost": 1, "difficulti": 1, "12": [1, 2], "fileset": 1, "indic": 1, "train": 1, "test": 1, "set": [1, 2], "13": 1, "mass_shap": 1, "14": 1, "mass_margin": 1, "separ": 1, "from": 1, "adjac": 1, "parenchyma": 1, "15": 1, "case_id": 1, "16": [1, 2], "whether": 1, "diagnos": 1, "true": 1, "fals": 1, "series_uid": 1, "seri": 1, "filepath": 1, "path": 1, "photometric_interpret": 1, "intend": 1, "samples_per_pixel": 1, "plane": 1, "row": 1, "column": 1, "aspect_ratio": 1, "continu": 1, "vertic": 1, "horizont": 1, "bit_depth": 1, "store": 1, "min_pixel_valu": 1, "minimum": 1, "actual": 1, "encount": 1, "largest_image_pixel": 1, "maximum": 1, "range_pixel_valu": 1, "differ": 1, "between": 1, "largest": 1, "smallest": 1, "brisqu": 1, "score": 1, "17": 1, "series_descript": 1, "full": 1, "far": 1, "better": 1, "approxim": 1, "answer": [1, 2], "often": 1, "vagu": 1, "than": [1, 2], "exact": 1, "wrong": 1, "can": 1, "alwai": 1, "made": 1, "precis": 1, "john": 1, "tukei": 1, "here": 1, "ll": 1, "put": 1, "forward": 1, "motiv": 1, "discoveri": 1, "gener": 1, "consid": 1, "potenti": 1, "signal": 1, "ar": [1, 2], "subtl": 1, "dens": [1, 2], "Is": 1, "less": 1, "To": 1, "what": [1, 2], "agreement": 1, "affect": 1, "most": 1, "do": 1, "impli": 1, "about": 1, "certain": 1, "other": 1, "support": 1, "term": 1, "consist": 1, "across": 1, "all": 1, "mammogram": [1, 2], "artifact": 1, "mark": 1, "text": 1, "extant": 1, "would": 1, "bright": 1, "contrast": 1, "visual": 1, "intra": 1, "class": 1, "dissimilar": 1, "inter": 1, "similar": 1, "factor": 1, "three": 1, "preliminari": 1, "ha": 1, "python": 1, "packag": 1, "depend": 1, "panda": 1, "tabular": 1, "numpi": 1, "numer": 1, "matplotlib": 1, "seaborn": 1, "scipi": 1, "statist": 1, "studioai": 1, "pd": 1, "stat": 1, "pickl": 1, "ipython": 1, "displai": 1, "display_html": 1, "html": 1, "np": 1, "pyplot": 1, "plt": 1, "sn": 1, "object": 1, "so": 1, "sklearn": 1, "svm": 1, "svc": 1, "linear_model": 1, "logisticregress": 1, "ensembl": 1, "randomforestclassifi": 1, "bcd": 1, "caseexplor": 1, "dicomexplor": 1, "pipelinebuild": 1, "modelselector": 1, "option": 1, "max_row": 1, "999": 1, "max_column": 1, "set_styl": 1, "whitegrid": 1, "set_palett": 1, "blues_r": 1, "modulenotfounderror": 1, "traceback": 1, "recent": 1, "call": 1, "last": 1, "cell": 1, "line": 1, "No": [1, 2], "modul": 1, "name": 1, "case_fp": 1, "calc": 1, "df": 1, "get_calc_data": 1, "get_mass_data": 1, "compris": 1, "analys": 1, "let": 1, "s": [1, 2], "get": 1, "sens": 1, "1566": 1, "3566": 1, "1872": 1, "1199": 1, "673": 1, "1694": 1, "910": 1, "784": 1, "st": 1, "t": 1, "pct_calc": 1, "round": 1, "100": 1, "pct_mass": 1, "pct_calc_mal": 1, "pct_calc_bn": 1, "pct_mass_mal": 1, "pct_mass_bn": 1, "cases_per_pati": 1, "msg": 1, "f": [1, 2], "kei": 1, "observ": 1, "n": 1, "tthe": 1, "comport": 1, "tcia": 1, "twe": 1, "have": 1, "tof": 1, "ton": 1, "averag": 1, "print": 1, "52": 1, "47": 1, "Of": 1, "64": 1, "05": [1, 2], "35": 1, "95": 1, "53": 1, "72": 1, "46": 1, "28": 1, "On": 1, "take": 1, "look": 1, "642": 1, "p_00806_left_calcification_mlo_1": 1, "p_00806": 1, "pleomorph": 1, "cluster": 1, "1549": 1, "p_00038_right_calcification_cc_2": 1, "p_00038": 1, "vascular": 1, "segment": 1, "1293": 1, "p_01679_right_calcification_mlo_1": 1, "p_01679": 1, "linear": 1, "248": 1, "p_00360_right_calcification_mlo_1": 1, "p_00360": 1, "546": 1, "p_00680_left_calcification_cc_1": 1, "p_00680": 1, "punctat": 1, "diffusely_scatt": 1, "3007": 1, "p_01638_right_mass_mlo_1": 1, "p_01638": 1, "lobul": 1, "obscur": [1, 2], "2108": 1, "p_00342_right_mass_mlo_3": 1, "p_00342": 1, "oval": 1, "circumscrib": [1, 2], "2321": 1, "p_00651_right_mass_mlo_1": 1, "p_00651": 1, "irregular": [1, 2], "ill_defin": 1, "3519": 1, "p_01645_right_mass_cc_1": 1, "p_01645": 1, "2385": 1, "p_00732_left_mass_mlo_1": 1, "p_00732": 1, "spicul": 1, "our": 1, "cover": 1, "And": 1, "begin": 1, "radiologist": 1, "classifi": 1, "us": [1, 2], "level": 1, "scale": 1, "cite": 1, "breastimagingreport": 1, "almost": 1, "entir": 1, "fatti": 1, "scatter": 1, "area": 1, "fibroglandular": 1, "heterogen": 1, "extrem": 1, "accord": 1, "american": 1, "colleg": 1, "radiolog": [1, 2], "u": 1, "women": 1, "40": 1, "figur": 1, "density_ref": 1, "exhibit_depth": 1, "count": 1, "vi": 1, "same": 1, "fig": 1, "ax": 1, "subplot": 1, "figsiz": 1, "plot": 1, "countplot": 1, "x": 1, "hue": 1, "titl": 1, "dodg": 1, "interest": 1, "pretti": 1, "close": 1, "unit": 1, "state": 1, "plot_count": 1, "normal": 1, "extent": 1, "point": 1, "make": 1, "balanc": 1, "respect": 1, "digit": [1, 2], "two": 1, "cranial": 1, "caudal": 1, "taken": 1, "abov": 1, "wherea": 1, "center": 1, "outward": 1, "slightli": 1, "greater": 1, "howev": 1, "reason": 1, "sequenc": 1, "assign": 1, "vast": 1, "present": 1, "singl": 1, "although": 1, "consider": 1, "common": 1, "especi": 1, "after": 1, "ag": 1, "50": 1, "calcium": 1, "deposit": 1, "within": 1, "typic": 1, "up": 1, "macrocalcif": 1, "microcalcif": 1, "appear": 1, "white": 1, "dot": 1, "dash": 1, "noncancer": 1, "requir": 1, "further": 1, "fine": 1, "speck": 1, "grain": 1, "salt": 1, "usual": 1, "pattern": 1, "earli": 1, "sign": 1, "also": 1, "particularli": 1, "reproduct": 1, "For": 1, "25": 1, "diseas": 1, "lifetim": 1, "initi": 1, "new": 1, "primari": 1, "care": 1, "wide": 1, "caus": 1, "physiolog": 1, "adenosi": 1, "aggress": [1, 2], "As": 1, "shown": 1, "below": 1, "measur": 1, "difficult": 1, "obviou": 1, "trend": 1, "toward": 1, "A": [1, 2], "definit": 1, "mean": 1, "find": 1, "unclear": 1, "need": 1, "neg": 1, "asymmetri": 1, "been": 1, "found": 1, "while": 1, "mai": [1, 2], "detect": [1, 2], "like": 1, "suspect": 1, "There": 1, "subcategori": 1, "4a": 1, "4b": 1, "4c": 1, "chanc": 1, "higher": 1, "being": 1, "previous": 1, "biopsi": 1, "describ": 1, "morpholog": 1, "differenti": 1, "over": 1, "yet": 1, "main": 1, "amorph": 1, "indistinct": 1, "without": 1, "clearli": 1, "defin": 1, "hazi": 1, "coars": 1, "conspicu": [1, 2], "larger": 1, "mm": 1, "dystroph": 1, "lava": 1, "year": 1, "treatment": 1, "30": 1, "eggshel": 1, "veri": 1, "thin": 1, "branch": 1, "curvilinear": 1, "rod": 1, "form": 1, "occassion": 1, "lucent": 1, "fat": 1, "necrosi": 1, "calcifi": 1, "debri": 1, "duct": 1, "milk": 1, "sediment": 1, "macro": 1, "microcyst": 1, "vari": 1, "skin": 1, "parallel": 1, "track": 1, "blood": 1, "vessel": 1, "y": 1, "order_by_count": 1, "diffus": 1, "throughout": 1, "whole": 1, "region": 1, "expect": 1, "ductal": 1, "group": 1, "least": 1, "lobe": 1, "lexicon": 1, "repres": 1, "low": 1, "undetermin": 1, "likelihood": 1, "microlobul": 1, "carcinoma": 1, "ill": 1, "frequent": 1, "distinguish": 1, "outcom": [1, 2], "callback": 1, "latter": 1, "should": 1, "monitor": 1, "investig": 1, "proport": 1, "collaps": 1, "sever": 1, "fall": 1, "one": 1, "five": 1, "similarli": 1, "nearli": 1, "20": 1, "next": 1, "former": 1, "upon": 1, "explanatori": 1, "independ": 1, "as_df": 1, "categorize_ordin": 1, "color": 1, "add": 1, "bar": 1, "stack": 1, "theme": 1, "axes_styl": 1, "grid": 1, "linestyl": 1, "label": 1, "layout": 1, "engin": 1, "tight": 1, "rather": 1, "prop": 1, "groupbi": 1, "value_count": 1, "to_fram": 1, "reset_index": 1, "sort_valu": 1, "risk": 1, "notwithstand": 1, "don": 1, "reveal": 1, "strong": 1, "infer": 1, "kt": 1, "kendallstau": 1, "b": [1, 2], "kendal": 1, "\u03c4": 1, "011138411035362646": 1, "pvalu": 1, "5382894688881223": 1, "alpha": 1, "strength": 1, "weak": 1, "tau": 1, "non": 1, "signific": 1, "phi_": 1, "01": 1, "p": [1, 2], "54": 1, "2022": [1, 2], "studi": 1, "publish": 1, "suggest": 1, "preval": 1, "bodi": 1, "If": 1, "evid": 1, "cv": 1, "cramersv": 1, "cramer": 1, "v": 1, "028861432861833916": 1, "08480010265447133": 1, "neglig": 1, "dof": 1, "x2alpha": 1, "x2": 1, "970414906184831": 1, "x2dof": 1, "chi": 1, "squar": 1, "97": 1, "08": 1, "phi": 1, "03": 1, "rsna": [1, 2], "journal": 1, "analyz": 1, "were": 1, "high": 1, "craniocaud": 1, "59": 1, "41": 1, "both": 1, "0014163311721585292": 1, "9325971883801198": 1, "007153374565586881": 1, "007": 1, "93": 1, "002": 1, "57": 1, "43": 1, "These": 1, "10437040964253724": 1, "5880686528820935e": 1, "38": 1, "84508847031938": 1, "85": 1, "compar": 1, "vs": 1, "report": [1, 2], "seven": 1, "incomplet": 1, "addit": 1, "evalu": 1, "comparison": 1, "na": 1, "routin": 1, "essenti": 1, "short": 1, "interv": 1, "month": 1, "known": 1, "proven": 1, "df1_style": 1, "style": 1, "set_table_attribut": 1, "inlin": 1, "220px": 1, "set_capt": 1, "df2_style": 1, "120px": 1, "_repr_html_": 1, "raw": 1, "nbsp": 1, "177": [1, 2], "375": 1, "102": 1, "902": 1, "731": 1, "560": 1, "750000": 1, "250000": 1, "000000": 1, "996894": 1, "003106": 1, "786164": 1, "213836": 1, "552358": 1, "447642": 1, "022688": 1, "977312": 1, "5994799138998625": 1, "4696313517612682e": 1, "244": 1, "inde": 1, "60": 1, "had": 1, "onli": 1, "At": 1, "sharp": 1, "increas": 1, "isn": 1, "clear": 1, "examin": 1, "seem": 1, "78": 1, "96": 1, "232": 1, "207": 1, "627": 1, "337": 1, "559": 1, "316": 1, "613": 1, "501": 1, "448276": 1, "551724": 1, "528474": 1, "471526": 1, "650415": 1, "349585": 1, "638857": 1, "361143": 1, "550269": 1, "449731": 1, "again": 1, "draw": 1, "003196827770471352": 1, "8618112089236021": 1, "003": [1, 2], "86": 1, "literatur": 1, "highest": 1, "concern": 1, "df_calc": 1, "5363368552127653": 1, "078377585363777e": 1, "87": 1, "538": 1, "4943200698192": 1, "42": 1, "539": 1, "69": [1, 2], "list": 1, "top": 1, "get_most_malignant_calc": 1, "barplot": 1, "32568564870672534": 1, "310406740502804e": 1, "39": 1, "198": 1, "5651774000304": 1, "56": 1, "33": 1, "df_mass": 1, "get_most_malignant_mass": 1, "510182454781321": 1, "297593104510473e": 1, "81": 1, "440": 1, "9247163603807": 1, "19": 1, "92": 1, "51": 1, "enabl": 1, "5894681913733985": 1, "1871720584088994e": 1, "113": 1, "588": 1, "6188361978973": 1, "18": 1, "62": 1, "That": 1, "conclud": 1, "exercis": 1, "start": 1, "Then": 1, "avoid": 1, "spuriou": 1, "plot_feature_associ": 1, "ignor": 1, "associationss": 1, "now": 1, "said": 1, "df_prop": 1, "sort": 1, "tend": 1, "behav": 1, "thu": 1, "plot_calc_feature_associ": 1, "strongli": 1, "closer": 1, "5399745903685791": 1, "2183": 1, "2953161289365": 1, "168": 1, "signfic": 1, "compound": 1, "therebi": 1, "reduc": 1, "_": 1, "summarize_morphology_by_featur": 1, "render": 1, "suspicion": 1, "those": 1, "intermedi": 1, "remain": 1, "classif": 1, "compare_morpholog": 1, "m1": 1, "m2": 1, "thei": 1, "co": 1, "occur": 1, "instanc": 1, "exclus": 1, "cours": 1, "large_rodlik": 1, "lucent_cent": 1, "regular": 1, "anyth": 1, "primarili": 1, "lastli": 1, "middl": 1, "rodlik": 1, "stand": 1, "specif": 1, "plot_mass_feature_associ": 1, "coupl": 1, "notabl": 1, "weakli": 1, "architectur": 1, "distort": 1, "perhap": 1, "compon": 1, "summar": 1, "part": 1, "pair": 1, "plot_target_associ": 1, "depict": 1, "mani": 1, "updat": 1, "gather": 1, "physician": 1, "strongest": 1, "rate": 1, "exceed": 1, "80": 1, "elucid": 1, "beyond": 1, "deriv": 1, "therefor": 1, "impact": 1, "estim": 1, "explain": 1, "establish": 1, "given": 1, "logist": 1, "regress": 1, "vector": 1, "machin": 1, "random": 1, "forest": 1, "cross": 1, "best": 1, "algorithm": 1, "built": 1, "pb": 1, "set_job": 1, "set_standard_scal": 1, "set_scor": 1, "accuraci": 1, "params_lr": 1, "clf__penalti": 1, "l1": 1, "l2": 1, "clf__c": 1, "clf__solver": 1, "liblinear": 1, "clf": 1, "random_st": 1, "set_classifi": 1, "param": 1, "build_gridsearch_cv": 1, "lr": 1, "params_svc": 1, "clf__kernel": 1, "param_rang": 1, "params_rf": 1, "clf__criterion": 1, "gini": 1, "entropi": 1, "clf__min_samples_leaf": 1, "clf__max_depth": 1, "clf__min_samples_split": 1, "rf": 1, "calc_train_fp": 1, "os": 1, "abspath": 1, "cook": 1, "calc_train": 1, "csv": 1, "calc_test_fp": 1, "calc_test": 1, "read_csv": 1, "x_train": 1, "loc": 1, "y_train": 1, "x_test": 1, "y_test": 1, "concat": 1, "axi": 1, "best_calc_model_fp": 1, "best_calc_pipelin": 1, "pkl": 1, "selector": 1, "calc_m": 1, "add_pipelin": 1, "run": 1, "forc": 1, "force_model_fit": 1, "y_pred": 1, "y_true": 1, "recal": 1, "f1": 1, "68": 1, "76": [1, 2], "58": 1, "79": 1, "67": 1, "avg": 1, "74": 1, "weight": 1, "73": 1, "outperform": 1, "achiev": 1, "posit": 1, "abil": 1, "harmon": 1, "coeffici": 1, "task": 1, "provid": 1, "belong": 1, "ncalcif": 1, "nfeatur": 1, "plot_feature_import": 1, "greatest": 1, "align": 1, "current": 1, "understood": 1, "mass_train_fp": 1, "mass_train": 1, "mass_test_fp": 1, "mass_test": 1, "best_mass_model_fp": 1, "best_mass_pipelin": 1, "mass_m": 1, "77": 1, "perform": 1, "nmass": 1, "unlik": 1, "against": 1, "comput": [1, 2], "its": 1, "puriti": 1, "leav": 1, "tree": 1, "pure": 1, "length": 1, "thick": 1, "radiat": 1, "demarc": 1, "lesion": 1, "surround": 1, "neither": 1, "nor": 1, "hidden": 1, "fibro": 1, "glandular": 1, "henc": 1, "cannot": 1, "fulli": 1, "practic": 1, "commonli": [1, 2], "when": 1, "portion": 1, "lower": 1, "avail": 1, "ds": 1, "dicom_fp": 1, "info": 1, "datatyp": 1, "complet": 1, "null": 1, "duplic": 1, "10703": 1, "00": 1, "6774": 1, "3929": 1, "63": 1, "1292410": 1, "10238": 1, "465": 1, "1814748": 1, "9137": 1, "684992": 1, "10701": 1, "10934": 1, "10822": 1, "10702": 1, "10771": 1, "int32": 1, "42812": 1, "int64": 1, "1077": 1, "9626": 1, "85624": 1, "1208": 1, "9495": 1, "5531": 1, "5172": 1, "float64": 1, "5591": 1, "5112": 1, "10820": 1, "3026": 1, "7677": 1, "235": 1, "10468": 1, "02": [1, 2], "3255": 1, "7448": 1, "10167": 1, "536": 1, "10700": 1, "791914": 1, "3568": 1, "7135": 1, "907015": 1, "10697": 1, "10699": 1, "10696": 1, "703253": 1, "21": [1, 2], "44": 1, "10659": 1, "759592": 1, "22": 1, "10694": 1, "732388": 1, "23": 1, "24": [1, 2], "721998": 1, "10698": 1, "26": 1, "661294": 1, "27": 1, "10682": 1, "737966": 1, "10683": 1, "749552": 1, "29": [1, 2], "359980": 1, "categor": 1, "uid": 1, "referenc": 1, "focu": 1, "represent": 1, "monochrome2": 1, "evenli": 1, "nrow": 1, "ncol": 1, "histogram": 1, "suptitl": 1, "nimag": 1, "tight_layout": 1, "transform": 1, "df_full": 1, "df_full_desc": 1, "datafram": 1, "smallest_pixel_valu": 1, "largest_pixel_valu": 1, "std": 1, "min": 1, "75": 1, "max": 1, "567": 1, "926": 1, "946": 1, "968": 1, "65": 1, "535": 1, "zero": 1, "rai": 1, "absorpt": 1, "anatomi": 1, "radiograph": 1, "beam": 1, "geometr": 1, "unsharp": 1, "resolut": 1, "annot": 1, "nois": 1, "record": 1, "system": 1, "blind": 1, "referenceless": 1, "spatial": [1, 2], "art": 1, "nr": 1, "iqa": 1, "boxplot": 1, "09": [1, 2], "82": 1, "116": 1, "gaussian": 1, "117": 1, "worst": 1, "preprocess": 1, "enhanc": 1, "inspect": 1, "interclass": 1, "intraclass": 1, "organ": 1, "variou": 1, "plot_imag": 1, "thirteen": 1, "remov": 1, "condit": 1, "lambda": 1, "isin": 1, "breast": 2, "url": 2, "http": 2, "www": 2, "ca": 2, "articl": 2, "visit": 2, "2023": 2, "04": 2, "ask": 2, "question": 2, "nci": 2, "februari": 2, "2018": 2, "archiv": 2, "locat": 2, "nciglob": 2, "ncienterpris": 2, "cancer": 2, "gov": 2, "chang": 2, "yara": 2, "abdou": 2, "medhavi": 2, "gupta": 2, "mariko": 2, "asaoka": 2, "kristoph": 2, "attwood": 2, "opyrch": 2, "mateusz": 2, "shipra": 2, "gandhi": 2, "kazuaki": 2, "takab": 2, "left": 2, "side": 2, "associ": 2, "biologi": 2, "wors": 2, "right": 2, "scientif": 2, "13377": 2, "august": 2, "com": 2, "s41598": 2, "022": 2, "16749": 2, "doi": 2, "1038": 2, "katrina": 2, "e": 2, "korhonen": 2, "emili": 2, "conant": 2, "eric": 2, "cohen": 2, "mari": 2, "synnestvedt": 2, "elizabeth": 2, "mcdonald": 2, "susan": 2, "weinstein": 2, "simultan": 2, "acquir": 2, "mammograph": 2, "imag": 2, "versu": 2, "tomosynthesi": 2, "292": 2, "juli": 2, "2019": 2, "pub": 2, "org": 2, "1148": 2, "radiol": 2, "2019182027": 2, "lawrenc": 2, "w": 2, "bassett": 2, "karen": 2, "conner": 2, "iv": 2, "ms": 2, "The": 2, "abnorm": 2, "holland": 2, "frei": 2, "medicin": 2, "6th": 2, "edit": 2, "bc": 2, "decker": 2, "2003": 2, "ncbi": 2, "nlm": 2, "nih": 2, "nbk12642": 2, "rebecca": 2, "sawyer": 2, "lee": 2, "francisco": 2, "gimenez": 2, "assaf": 2, "hoogi": 2, "kana": 2, "kawai": 2, "miyak": 2, "mia": 2, "gorovoi": 2, "daniel": 2, "l": 2, "rubin": 2, "curat": 2, "aid": 2, "diagnosi": 2, "research": 2, "170177": 2, "decemb": 2, "2017": 2, "sdata2017177": 2, "sdata": 2, "david": 2, "g": 2, "bragg": 2, "philip": 2, "hedvig": 2, "hricak": 2, "oncolog": 2, "saunder": 2, "philadelphia": 2, "2nd": 2, "ed": 2, "2002": 2, "isbn": 2, "978": 2, "7216": 2, "7494": 2, "erin": 2, "princ": 2, "heidi": 2, "r": 2, "umphrei": 2, "multipl": 2, "mass": 2, "oxford": 2, "univers": 2, "press": 2, "march": 2, "academ": 2, "oup": 2, "24629": 2, "chapter": 2, "187959266": 2, "1093": 2, "med": 2, "9780190270261": 2, "0019": 2, "187965422": 2, "0026": 2, "elezabi": 2, "187962148": 2, "0023": 2, "mittal": 2, "k": 2, "moorthi": 2, "c": 2, "bovik": 2, "qualiti": 2, "assess": 2, "domain": 2, "ieee": 2, "transact": 2, "4695": 2, "4708": 2, "2012": 2, "ieeexplor": 2, "6272356": 2, "1109": 2, "tip": 2, "2214050": 2}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"exploratori": 1, "data": 1, "analysi": 1, "eda": 1, "cbi": 1, "ddsm": 1, "dataset": 1, "The": 1, "case": 1, "dictionari": 1, "dicom": 1, "approach": 1, "guid": 1, "question": 1, "plan": 1, "univari": 1, "breast": 1, "densiti": 1, "left": 1, "right": 1, "side": 1, "imag": 1, "view": 1, "abnorm": 1, "id": 1, "type": 1, "subtleti": 1, "bi": 1, "rad": 1, "assess": 1, "calcif": 1, "distribut": 1, "mass": 1, "shape": 1, "margin": 1, "patholog": 1, "cancer": 1, "summari": 1, "bivari": 1, "target": 1, "variabl": 1, "associ": 1, "diagnosi": 1, "featur": 1, "larg": 1, "effect": 1, "moder": 1, "multivari": 1, "build": 1, "pipelin": 1, "load": 1, "execut": 1, "model": 1, "select": 1, "predict": 1, "result": 1, "import": 1, "metadata": 1, "photometr": 1, "interpret": 1, "sampl": 1, "per": 1, "pixel": 1, "bit": 1, "depth": 1, "height": 1, "width": 1, "size": 1, "aspect": 1, "ratio": 1, "valu": 1, "qualiti": 1, "benign": 1, "probabl": 1, "suspici": 1, "highli": 1, "refer": 2}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})
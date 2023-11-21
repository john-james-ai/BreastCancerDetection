Search.setIndex({"docnames": ["content/00_intro", "content/07_eda", "content/09_image_prep", "content/99_references"], "filenames": ["content/00_intro.md", "content/07_eda.ipynb", "content/09_image_prep.md", "content/99_references.md"], "titles": ["&lt;no title&gt;", "Exploratory Data Analysis (EDA) of the <strong>CBIS-DDSM</strong> Dataset", "Image Preprocessing", "References"], "terms": {"thi": [0, 1, 2], "small": [0, 1, 2], "sampl": [0, 2], "book": [0, 3], "give": [0, 2], "you": 0, "feel": 0, "how": [0, 1, 2], "content": 0, "structur": [0, 2], "It": [0, 1, 2], "show": [0, 1, 2], "off": 0, "few": [0, 1, 2], "major": [0, 1], "file": [0, 1], "type": [0, 3], "well": [0, 1, 2], "some": [0, 1, 2], "doe": [0, 1], "go": 0, "depth": 0, "ani": [0, 1, 2], "particular": [0, 1], "topic": 0, "check": [0, 1], "out": [0, 1, 2], "jupyt": 0, "document": [0, 3], "more": [0, 1, 2], "inform": [0, 1, 2], "page": 0, "bundl": 0, "see": [0, 1, 2], "exploratori": [0, 2], "data": [0, 2, 3], "analysi": [0, 2], "eda": 0, "cbi": [0, 2], "ddsm": [0, 2], "dataset": [0, 2], "imag": [0, 3], "preprocess": [0, 1], "refer": [0, 1, 2], "In": [1, 2, 3], "section": 1, "we": [1, 2], "conduct": 1, "an": [1, 2, 3], "prepar": 1, "prior": 1, "purpos": 1, "three": 1, "fold": 1, "discov": 1, "relationship": 1, "among": 1, "explor": 1, "natur": [1, 2, 3], "between": [1, 2], "diagnost": [1, 2], "properti": [1, 2], "thei": [1, 2], "pertain": 1, "involv": [1, 2], "follow": [1, 2], "contain": [1, 2], "patient": 1, "image_view": 1, "i": [1, 2, 3], "e": [1, 2, 3], "mammographi": [1, 3], "roi": [1, 2], "mask": [1, 2], "crop": 1, "format": 1, "descript": 1, "1": [1, 2, 3], "patient_id": 1, "nomin": 1, "uniqu": 1, "identifi": 1, "each": [1, 2], "2": [1, 2, 3], "breast_dens": 1, "discret": [1, 2], "overal": [1, 2], "volum": [1, 3], "attenu": 1, "tissu": [1, 2], "3": [1, 2, 3], "left_or_right_breast": 1, "which": [1, 2], "wa": 1, "4": [1, 2, 3], "dichotom": 1, "either": [1, 2], "cranialcaud": 1, "mediolater": 1, "obliqu": 1, "5": [1, 2, 3], "abnormality_id": 1, "number": [1, 2], "6": [1, 2, 3], "abnormality_typ": 1, "categori": 1, "7": [1, 2, 3], "calc_typ": 1, "character": [1, 2], "where": [1, 2], "applic": [1, 2], "8": [1, 2, 3], "calc_distribut": 1, "arrang": 1, "insid": 1, "rel": [1, 2], "malign": [1, 2], "9": [1, 3], "mass_shap": 1, "10": [1, 2, 3], "mass_margin": 1, "separ": [1, 2], "from": [1, 2], "adjac": 1, "parenchyma": 1, "11": [1, 3], "12": [1, 2, 3], "determin": 1, "13": [1, 2, 3], "degre": 1, "difficulti": 1, "14": [1, 2, 3], "fileset": 1, "indic": 1, "train": [1, 2], "test": [1, 2], "set": [1, 3], "15": [1, 2, 3], "case_id": 1, "16": [1, 2, 3], "whether": 1, "diagnos": 1, "As": 1, "describ": [1, 2], "compound": 1, "morpholog": [1, 2], "were": 1, "unari": 1, "dummi": 1, "encod": 1, "series_uid": 1, "seri": [1, 3], "filepath": 1, "path": 1, "photometric_interpret": 1, "intend": 1, "samples_per_pixel": 1, "plane": 1, "row": 1, "column": 1, "aspect_ratio": 1, "continu": [1, 2], "vertic": 1, "horizont": 1, "bit_depth": 1, "store": 1, "min_pixel_valu": 1, "minimum": [1, 2], "actual": 1, "encount": [1, 2], "largest_image_pixel": 1, "maximum": [1, 2], "range_pixel_valu": 1, "differ": 1, "largest": [1, 2], "smallest": 1, "series_descript": 1, "full": [1, 2, 3], "also": [1, 2], "includ": [1, 2], "far": 1, "better": 1, "approxim": [1, 2, 3], "answer": [1, 3], "often": 1, "vagu": 1, "than": [1, 2, 3], "exact": 1, "wrong": 1, "can": [1, 2], "alwai": 1, "made": 1, "precis": [1, 2], "john": 1, "tukei": 1, "here": [1, 2], "ll": 1, "put": 1, "forward": 1, "motiv": 1, "discoveri": [1, 2], "process": [1, 2, 3], "what": [1, 2], "ar": [1, 2], "To": [1, 2], "relat": [1, 2], "certain": 1, "less": [1, 2], "subtl": [1, 2], "concern": 1, "do": [1, 2], "have": [1, 2], "standard": [1, 2], "represent": [1, 2], "term": 1, "artifact": [1, 2], "mark": 1, "text": [1, 2], "extant": 1, "would": [1, 2], "bright": [1, 2], "contrast": [1, 2], "primari": 1, "stage": [1, 2], "preliminari": 1, "ha": [1, 2], "python": 1, "packag": 1, "depend": [1, 2], "panda": 1, "tabular": 1, "numpi": 1, "numer": [1, 2], "matplotlib": 1, "seaborn": 1, "visual": 1, "scipi": 1, "statist": [1, 2, 3], "studioai": 1, "pd": 1, "stat": 1, "pickl": 1, "np": 1, "pyplot": 1, "plt": 1, "sn": [1, 2], "object": 1, "so": 1, "sklearn": 1, "svm": 1, "svc": 1, "linear_model": 1, "logisticregress": 1, "ensembl": 1, "randomforestclassifi": 1, "bcd": 1, "analyz": 1, "caseexplor": 1, "dicomexplor": 1, "pipelinebuild": 1, "modelselector": 1, "option": 1, "displai": 1, "max_row": 1, "999": 1, "max_column": 1, "set_styl": 1, "whitegrid": 1, "set_palett": 1, "blues_r": 1, "case_fp": 1, "calc": 1, "df": 1, "get_calc_data": 1, "get_mass_data": 1, "compris": 1, "analys": 1, "let": [1, 2], "s": [1, 2, 3], "sens": [1, 2], "1566": 1, "3566": 1, "1872": 1, "1199": 1, "673": 1, "1694": 1, "910": 1, "784": 1, "st": 1, "t": [1, 2], "pct_calc": 1, "round": 1, "100": 1, "0": [1, 2, 3], "pct_mass": 1, "pct_calc_mal": 1, "pct_calc_bn": 1, "pct_mass_mal": 1, "pct_mass_bn": 1, "cases_per_pati": 1, "msg": 1, "f": [1, 2, 3], "kei": 1, "observ": [1, 2], "n": [1, 2, 3], "tthe": 1, "comport": 1, "tcia": 1, "twe": 1, "tof": 1, "ton": 1, "averag": [1, 2], "print": 1, "52": 1, "47": 1, "Of": 1, "64": 1, "05": [1, 3], "35": 1, "95": 1, "53": 1, "72": 1, "46": 1, "28": 1, "On": [1, 3], "take": [1, 2], "look": 1, "1746": 1, "test_p_01308_left_mlo_1": 1, "p_01308": 1, "mlo": 1, "pleomorph": 1, "cluster": 1, "fals": [1, 2], "913": 1, "train_p_01166_right_mlo_1": 1, "p_01166": 1, "true": [1, 2], "509": 1, "train_p_00635_right_mlo_1": 1, "p_00635": 1, "lucent_cent": 1, "segment": [1, 2], "benign_without_callback": 1, "1033": 1, "train_p_01321_right_mlo_3": 1, "p_01321": 1, "fine_linear_branch": 1, "linear": [1, 2], "1692": 1, "test_p_00919_right_cc_1": 1, "p_00919": 1, "cc": 1, "3448": 1, "test_p_01277_right_mlo_1": 1, "p_01277": 1, "irregular": [1, 2, 3], "ill_defin": 1, "2083": 1, "train_p_00314_right_cc_1": 1, "p_00314": 1, "lobul": 1, "microlobul": 1, "2634": 1, "train_p_01112_left_mlo_1": 1, "p_01112": 1, "architectural_distort": 1, "spicul": 1, "2157": 1, "train_p_00417_right_mlo_1": 1, "p_00417": 1, "3128": 1, "train_p_01814_right_cc_1": 1, "p_01814": 1, "obscur": [1, 3], "our": 1, "cover": 1, "radiologist": 1, "classifi": [1, 2], "us": [1, 2, 3], "level": [1, 2], "scale": 1, "almost": [1, 2], "entir": 1, "fatti": 1, "scatter": [1, 2], "area": [1, 2], "fibroglandular": 1, "heterogen": 1, "dens": [1, 2, 3], "extrem": 1, "note": [1, 2], "correspond": [1, 2], "b": [1, 2, 3], "c": [1, 2, 3], "d": [1, 2], "list": 1, "abov": 1, "confus": 1, "notwithstand": 1, "ordin": 1, "chart": 1, "illustr": [1, 2], "within": [1, 2], "fig": [1, 2], "ax": 1, "subplot": 1, "figsiz": 1, "plot": 1, "countplot": 1, "x": [1, 2], "titl": 1, "plot_count": 1, "balanc": 1, "respect": [1, 2], "digit": [1, 2, 3], "two": [1, 2], "cranial": 1, "caudal": 1, "taken": [1, 2], "best": 1, "subarcolar": 1, "central": [1, 2], "medial": 1, "posteromedi": 1, "project": 1, "its": [1, 2], "entireti": 1, "posterior": 1, "upper": 1, "outer": 1, "quadrant": 1, "proport": [1, 2], "sequenc": 1, "assign": 1, "count": [1, 2], "vast": 1, "present": [1, 2], "singl": 1, "although": 1, "consider": [1, 2], "common": [1, 2], "mammogram": [1, 2, 3], "especi": 1, "after": [1, 2], "ag": 1, "50": 1, "calcium": 1, "deposit": 1, "typic": [1, 2], "up": 1, "macrocalcif": 1, "microcalcif": 1, "appear": [1, 2], "larg": [1, 2], "white": [1, 2], "dot": 1, "dash": 1, "noncancer": 1, "requir": [1, 2], "further": 1, "fine": [1, 2], "speck": 1, "similar": [1, 2], "grain": 1, "salt": 1, "usual": [1, 2], "pattern": [1, 2, 3], "earli": 1, "sign": 1, "particularli": 1, "women": 1, "reproduct": 1, "For": [1, 2], "25": 1, "affect": [1, 2], "diseas": 1, "lifetim": 1, "initi": 1, "new": [1, 3], "care": [1, 2], "wide": 1, "rang": [1, 2], "caus": [1, 2], "physiolog": 1, "adenosi": 1, "aggress": [1, 3], "shown": 1, "below": 1, "measur": [1, 2], "difficult": 1, "obviou": 1, "17": [1, 3], "A": [1, 2, 3], "plural": 1, "moder": 1, "slightli": [1, 2], "nearli": 1, "3rd": 1, "consid": 1, "base": [1, 2], "upon": [1, 2], "thorough": 1, "evalu": [1, 2], "mammograph": [1, 2, 3], "six": 1, "definit": [1, 2], "mean": 1, "find": 1, "unclear": 1, "need": 1, "score": 1, "neg": 1, "normal": [1, 2], "No": 1, "asymmetri": 1, "other": [1, 2], "been": [1, 2], "found": 1, "while": [1, 2], "mai": [1, 2, 3], "detect": [1, 2, 3], "most": [1, 2], "like": 1, "suspect": 1, "four": 1, "subcategori": 1, "4a": 1, "4b": 1, "4c": 1, "chanc": 1, "higher": [1, 2], "being": [1, 2], "previous": 1, "biopsi": 1, "factor": [1, 2], "differenti": 1, "There": 1, "over": [1, 2], "40": 1, "main": 1, "amorph": 1, "indistinct": 1, "without": 1, "clearli": 1, "defin": [1, 2], "hazi": 1, "coars": 1, "conspicu": [1, 2, 3], "larger": [1, 2], "mm": 1, "dystroph": 1, "lava": 1, "develop": [1, 2], "year": 1, "treatment": 1, "about": 1, "30": 1, "eggshel": 1, "veri": 1, "thin": 1, "branch": 1, "curvilinear": 1, "rod": 1, "form": [1, 2], "occassion": 1, "lucent": 1, "center": [1, 2], "oval": 1, "fat": 1, "necrosi": 1, "calcifi": 1, "debri": 1, "duct": 1, "milk": 1, "sediment": 1, "macro": 1, "microcyst": 1, "vari": 1, "punctat": 1, "skin": 1, "vascular": 1, "parallel": 1, "track": 1, "blood": 1, "vessel": 1, "y": [1, 2], "order_by_count": 1, "account": 1, "half": 1, "75": 1, "repres": [1, 2], "five": [1, 2], "diffus": 1, "throughout": 1, "whole": 1, "region": [1, 2], "expect": [1, 2], "ductal": 1, "group": 1, "least": [1, 3], "lobe": 1, "80": [1, 2, 3], "calfic": 1, "lexicon": 1, "howev": [1, 2], "addit": 1, "symmetri": 1, "architectur": 1, "circumscrib": [1, 3], "low": [1, 2], "undetermin": 1, "likelihood": 1, "carcinoma": 1, "ill": 1, "call": [1, 2], "gener": [1, 2], "make": [1, 2], "70": 1, "distinguish": [1, 2], "outcom": [1, 3], "callback": 1, "latter": 1, "should": 1, "monitor": 1, "investig": 1, "collaps": 1, "sever": [1, 2], "fall": 1, "one": [1, 2], "similarli": [1, 2], "20": [1, 2], "yet": [1, 2], "class": [1, 2], "next": [1, 2], "inter": 1, "former": 1, "explanatori": 1, "independ": [1, 2], "as_df": 1, "categorize_ordin": 1, "color": 1, "add": 1, "bar": 1, "stack": 1, "theme": 1, "axes_styl": 1, "grid": 1, "linestyl": 1, "label": 1, "layout": 1, "engin": 1, "tight": 1, "rather": 1, "prop": 1, "groupbi": 1, "value_count": 1, "to_fram": 1, "reset_index": 1, "sort_valu": 1, "risk": 1, "don": 1, "reveal": [1, 2], "strong": 1, "support": 1, "infer": 1, "kt": 1, "kendallstau": 1, "name": 1, "kendal": 1, "\u03c4": 1, "011138411035362646": 1, "pvalu": 1, "5382894688881223": 1, "alpha": [1, 2], "strength": 1, "weak": 1, "tau": [1, 3], "non": [1, 2, 3], "signific": [1, 2], "effect": [1, 2], "phi_": 1, "01": [1, 2], "p": [1, 2, 3], "54": 1, "2022": [1, 3], "studi": 1, "publish": 1, "suggest": 1, "preval": 1, "bodi": [1, 2], "If": 1, "greater": [1, 2], "evid": 1, "cv": 1, "cramersv": 1, "cramer": 1, "v": 1, "028861432861833916": 1, "08480010265447133": 1, "neglig": 1, "dof": 1, "x2alpha": 1, "x2": 1, "970414906184831": 1, "x2dof": 1, "chi": 1, "squar": [1, 2, 3], "97": 1, "08": 1, "phi": 1, "03": [1, 3], "rsna": [1, 3], "journal": [1, 3], "high": [1, 2], "craniocaud": 1, "59": 1, "41": 1, "both": [1, 2], "same": [1, 2], "0014163311721585292": 1, "9325971883801198": 1, "007153374565586881": 1, "007": 1, "93": 1, "002": 1, "Is": 1, "57": 1, "43": 1, "These": 1, "10437040964253724": 1, "5880686528820935e": 1, "38": [1, 3], "84508847031938": 1, "85": 1, "compar": 1, "vs": 1, "di": 1, "agreement": 1, "report": [1, 3], "seven": 1, "incomplet": 1, "comparison": 1, "na": 1, "routin": 1, "essenti": 1, "short": 1, "interv": [1, 2], "month": 1, "known": [1, 2], "proven": 1, "concat": 1, "axi": 1, "177": [1, 3], "00": 1, "642": 1, "375": 1, "79": 1, "102": 1, "21": 1, "902": 1, "55": 1, "731": 1, "45": 1, "02": [1, 3], "560": 1, "98": 1, "5994799138998625": 1, "4696313517612682e": 1, "244": 1, "inde": 1, "60": [1, 3], "had": 1, "onli": [1, 2], "all": [1, 2], "ultim": 1, "just": [1, 2], "isn": 1, "clear": [1, 2], "examin": [1, 2], "vi": 1, "seem": 1, "78": 1, "96": 1, "232": 1, "207": 1, "627": 1, "65": [1, 3], "337": 1, "559": 1, "316": 1, "36": 1, "613": 1, "501": 1, "again": 1, "draw": 1, "003196827770471352": 1, "8618112089236021": 1, "003": [1, 3], "86": 1, "accord": 1, "literatur": 1, "highest": 1, "df_calc": 1, "5363368552127653": 1, "078377585363777e": 1, "87": [1, 3], "538": 1, "4943200698192": 1, "42": 1, "539": 1, "69": [1, 3], "top": 1, "get_most_malignant_calc": 1, "barplot": 1, "32634130163729136": 1, "693260459198279e": 1, "39": 1, "199": 1, "36546372889003": 1, "198": 1, "56": 1, "33": 1, "df_mass": 1, "get_most_malignant_mass": 1, "510182454781321": 1, "297593104510473e": 1, "81": 1, "440": 1, "9247163603807": 1, "19": [1, 3], "92": 1, "51": 1, "enabl": 1, "5894681913733985": 1, "1871720584088994e": 1, "113": 1, "588": 1, "6188361978973": 1, "18": [1, 3], "62": 1, "That": 1, "conclud": 1, "impli": 1, "exercis": 1, "start": 1, "Then": 1, "avoid": 1, "spuriou": 1, "across": 1, "plot_feature_associ": 1, "ignor": 1, "associationss": 1, "now": 1, "said": 1, "df_prop": 1, "sort": 1, "hue": 1, "tend": 1, "behav": 1, "thu": 1, "plot_calc_feature_associ": 1, "_": 1, "summarize_morphology_by_featur": 1, "render": 1, "suspicion": 1, "those": 1, "intermedi": 1, "remain": 1, "classif": [1, 2], "5399745903685791": 1, "2183": 1, "2953161289365": 1, "168": 1, "signfic": 1, "therebi": 1, "reduc": [1, 2], "compare_morpholog": 1, "m1": 1, "m2": 1, "co": 1, "occur": 1, "instanc": [1, 2], "exclus": 1, "45403619931007766": 1, "3087": 1, "2854813722943": 1, "336": 1, "cours": [1, 2], "large_rodlik": 1, "regular": [1, 2], "32542688359496785": 1, "164653782848766e": 1, "82": 1, "792": 1, "9990923686997": 1, "strongli": 1, "793": 1, "anyth": 1, "primarili": 1, "38789958530499524": 1, "1562872500601785e": 1, "216": 1, "1126": 1, "690069039047": 1, "32": 1, "1127": 1, "middl": 1, "rodlik": 1, "stand": 1, "specif": [1, 2], "3552474787413507": 1, "624419427603362e": 1, "708": 1, "7435307901172": 1, "126": 1, "obser": 1, "709": 1, "plot_mass_feature_associ": 1, "asses": 1, "37": 1, "notabl": 1, "weakli": 1, "perhap": 1, "40037910751937816": 1, "3306037914870686e": 1, "225": 1, "1357": 1, "7700498809766": 1, "90": 1, "1358": 1, "distort": 1, "3693905740902926": 1, "3746514669523e": 1, "182": 1, "1155": 1, "7263860406229": 1, "1156": 1, "compon": [1, 2], "summar": [1, 2], "part": [1, 2], "pair": 1, "plot_target_associ": 1, "depict": 1, "mani": [1, 2], "updat": 1, "gather": 1, "physician": 1, "strongest": 1, "rate": 1, "exceed": 1, "elucid": 1, "beyond": [1, 2], "deriv": 1, "therefor": 1, "impact": 1, "estim": 1, "explain": 1, "establish": 1, "given": [1, 2], "logist": 1, "regress": 1, "vector": 1, "machin": 1, "random": [1, 2], "forest": 1, "cross": 1, "valid": 1, "algorithm": [1, 2, 3], "built": 1, "pb": 1, "set_job": 1, "set_standard_scal": 1, "set_scor": 1, "accuraci": [1, 2], "params_lr": 1, "clf__penalti": 1, "l1": 1, "l2": 1, "clf__c": 1, "clf__solver": 1, "liblinear": 1, "clf": 1, "random_st": 1, "set_classifi": 1, "param": 1, "build_gridsearch_cv": 1, "lr": 1, "params_svc": 1, "clf__kernel": 1, "param_rang": 1, "params_rf": 1, "clf__criterion": 1, "gini": 1, "entropi": 1, "clf__min_samples_leaf": 1, "clf__max_depth": 1, "clf__min_samples_split": 1, "rf": 1, "x_train": 1, "y_train": 1, "x_test": 1, "y_test": 1, "get_calc_model_data": 1, "best_calc_model_fp": 1, "os": 1, "abspath": 1, "best_calc_pipelin": 1, "pkl": 1, "selector": 1, "calc_m": 1, "add_pipelin": 1, "run": 1, "forc": 1, "force_model_fit": 1, "y_pred": 1, "y_true": 1, "load": 1, "recal": 1, "f1": 1, "73": 1, "83": [1, 3], "avg": 1, "66": 1, "67": 1, "weight": 1, "76": [1, 3], "71": 1, "outperform": 1, "achiev": 1, "posit": 1, "abil": 1, "58": 1, "harmon": 1, "coeffici": 1, "task": [1, 2], "provid": [1, 2], "belong": 1, "wherea": [1, 2], "ncalcif": 1, "nfeatur": 1, "plot_feature_import": 1, "greatest": 1, "align": 1, "current": 1, "understood": 1, "get_mass_model_data": 1, "best_mass_model_fp": 1, "best_mass_pipelin": 1, "mass_m": 1, "77": 1, "84": 1, "perform": [1, 2], "nmass": 1, "unlik": 1, "against": 1, "comput": [1, 2, 3], "increas": [1, 2], "puriti": 1, "leav": 1, "tree": 1, "pure": 1, "point": 1, "line": [1, 2], "length": 1, "thick": 1, "radiat": [1, 2], "sharp": [1, 2], "demarc": 1, "lesion": [1, 2], "surround": 1, "neither": [1, 2], "nor": [1, 2], "hidden": 1, "fibro": 1, "glandular": 1, "henc": 1, "cannot": 1, "fulli": 1, "practic": [1, 3], "commonli": [1, 2, 3], "when": [1, 2], "portion": 1, "lower": 1, "avail": [1, 2], "ds": 1, "dicom_fp": 1, "info": 1, "datatyp": 1, "complet": 1, "null": 1, "duplic": 1, "uid": 1, "3565": 1, "3100": 1, "465": 1, "331545": 1, "430492": 1, "606875": 1, "3564": 1, "3633": 1, "int32": 1, "14260": 1, "int64": 1, "349": 1, "3216": 1, "28520": 1, "425": 1, "3140": 1, "2591": 1, "974": 1, "float64": 1, "2595": 1, "970": 1, "3624": 1, "max_pixel_valu": 1, "233": 1, "3332": 1, "07": 1, "mean_pixel_valu": 1, "3031": 1, "534": 1, "median_pixel_valu": 1, "1086": 1, "2479": 1, "std_pixel_valu": 1, "278070": 1, "1999": 1, "44": 1, "228160": 1, "3561": 1, "3563": 1, "3796": 1, "3684": 1, "3558": 1, "22": 1, "234313": 1, "23": 1, "3521": 1, "253086": 1, "24": [1, 3], "3555": 1, "244027": 1, "3559": 1, "26": 1, "3562": 1, "240560": 1, "27": 1, "3560": 1, "220326": 1, "29": [1, 3], "3544": 1, "245884": 1, "3545": 1, "249740": 1, "31": 1, "322810": 1, "bool": 1, "categor": 1, "referenc": 1, "focu": [1, 2, 3], "monochrome2": 1, "evenli": 1, "df_full": 1, "loc": [1, 3], "nrow": 1, "ncol": 1, "histogram": [1, 2], "suptitl": 1, "nimag": 1, "tight_layout": 1, "transform": [1, 2], "interest": [1, 2], "df_full_desc": 1, "datafram": 1, "std": 1, "min": [1, 2], "max": [1, 2], "565": 1, "926": 1, "946": 1, "968": 1, "535": 1, "zero": 1, "inspect": 1, "nois": 1, "annot": [1, 2], "interclass": 1, "dissimilar": 1, "intraclass": 1, "organ": 1, "variou": [1, 2], "plot_imag": 1, "thirteen": 1, "remov": [1, 2], "condit": [1, 2], "lambda": [1, 2], "isin": 1, "accur": 2, "diagnosi": [2, 3], "breast": [2, 3], "cancer": [2, 3], "rest": 2, "discriminatori": 2, "power": 2, "mathemat": [2, 3], "design": 2, "abnorm": [2, 3], "biomed": 2, "advanc": 2, "artifici": 2, "intellig": 2, "vision": [2, 3], "fuel": 2, "explos": 2, "ai": 2, "rise": 2, "recognit": [2, 3], "capabl": 2, "increasingli": 2, "complex": 2, "still": 2, "clinic": [2, 3], "qualiti": 2, "resolut": 2, "free": 2, "illumin": 2, "issu": [2, 3], "compromis": 2, "resembl": 2, "pixel": 2, "intens": 2, "interfer": 2, "extract": 2, "lead": 2, "obstacl": 2, "featur": 2, "poor": 2, "influenc": 2, "conceal": 2, "import": 2, "tumor": 2, "shape": 2, "ambigu": 2, "blur": 2, "edg": 2, "complic": 2, "deep": [2, 3], "learn": [2, 3], "The": [2, 3], "3500": 2, "address": 2, "challeng": 2, "fundament": 2, "regard": 2, "approach": [2, 3], "devis": 2, "elimin": 2, "optim": [], "produc": 2, "collect": 2, "maxim": 2, "begin": 2, "method": 2, "appli": 2, "onc": 2, "hyper": 2, "paramet": 2, "select": 2, "move": 2, "binar": 2, "threshold": 2, "pector": 2, "muscl": 2, "techniqu": 2, "canni": 2, "hough": 2, "contour": 2, "dure": 2, "enhanc": 2, "gamma": 2, "correct": 2, "limit": 2, "adapt": [2, 3], "equal": 2, "clahe": 2, "awgn": 2, "ad": 2, "improv": 2, "neural": 2, "network": [2, 3], "mitig": 2, "overfit": 2, "final": 2, "autom": 2, "creat": 2, "binari": 2, "inher": 2, "noisi": 2, "acquisit": 2, "transmiss": 2, "inject": [], "unwant": 2, "signal": [2, 3], "must": [], "minim": [], "befor": 2, "downstream": [], "discrep": 2, "amount": 2, "light": 2, "s_i": 2, "valu": 2, "x_i": 2, "subsect": [], "aim": 2, "function": 2, "approx": 2, "input": 2, "return": 2, "clean": 2, "output": 2, "critic": [], "knowledg": [], "manifest": [], "distribut": 2, "assess": [], "potenti": [], "solut": [], "might": 2, "smooth": 2, "formal": [], "constrain": [], "possibl": [], "probabl": 2, "broadli": 2, "speak": 2, "multipl": 3, "undesir": 2, "aris": 2, "get": 2, "theori": 2, "coordin": 2, "unobserv": [], "determinist": [], "corrupt": 2, "ident": [], "varianc": 2, "sigma": 2, "2_n": [], "origin": 2, "wise": [], "sum": 2, "multipli": 2, "captur": 2, "time": 2, "storag": 2, "wherebi": 2, "space": [], "impuls": [], "ubiquit": [], "telecommun": [], "system": [2, 3], "princip": [], "sourc": 2, "imageri": [], "sensor": 2, "subject": [], "overh": [], "disturb": [], "extern": [], "channel": [], "variat": 2, "electr": 2, "assum": [], "characterist": [], "densiti": 2, "first": [], "introduc": 2, "french": [], "mathematician": [], "abraham": [], "de": [], "moivr": [], "second": [], "edit": 3, "1718": [], "hi": [], "doctrin": [], "later": [], "attribut": [], "karl": [], "friedrich": [], "gauss": 3, "german": [], "work": [], "connect": [], "express": [], "bivari": [], "isotrop": [], "circular": [], "g": [2, 3], "frac": 2, "pi": 2, "sigma_x": [], "sigma_i": [], "mu_x": [], "mu_i": [], "dimens": [], "deviat": 2, "context": [], "guassian": [], "analog": 2, "convers": 2, "adc": 2, "consist": 2, "step": 2, "spatial": 2, "amplitud": 2, "grei": [], "unavoid": 2, "aspect": 2, "infinit": 2, "bit": 2, "assumpt": 2, "uniformli": 2, "unless": 2, "enough": [], "dither": [], "pre": [], "scene": 2, "irradi": 2, "photon": 2, "incid": 2, "sinc": 2, "individu": 2, "treat": 2, "event": 2, "tempor": 2, "element": 2, "pr": 2, "k": 2, "he": [], "per": 2, "unit": 2, "uncertainti": 2, "var": 2, "grow": 2, "root": 2, "broad": 2, "fact": 2, "prerequisit": 2, "field": 2, "research": [2, 3], "devot": 2, "ratio": 2, "snr": 2, "audio": [2, 3], "video": [2, 3], "systemat": 2, "review": 2, "landscap": 2, "scope": 2, "effort": 2, "reduct": 2, "median": [2, 3], "kernel": 2, "variabl": 2, "window": 2, "size": 2, "local": [2, 3], "simpl": 2, "intuit": 2, "easi": 2, "implement": 2, "pass": 2, "basic": 2, "oper": 2, "simpli": 2, "replac": 2, "neighborhood": 2, "thought": 2, "convolut": 2, "around": 2, "notion": 2, "3x3": 2, "5x5": 2, "7x7": 2, "specifi": 2, "url": 3, "http": 3, "www": 3, "acr": 3, "org": 3, "resourc": 3, "bi": 3, "rad": 3, "visit": 3, "2023": 3, "09": 3, "shelli": 3, "lill\u00e9": 3, "wendi": 3, "marshal": [2, 3], "valeri": 3, "andolina": 3, "guid": 3, "wolter": 3, "kluwer": 3, "philadelphia": 3, "fourth": 3, "2019": 3, "isbn": 3, "978": 3, "4963": 3, "5202": 3, "oclc": 3, "1021062474": 3, "ask": 3, "question": 3, "nci": 3, "februari": 3, "2018": 3, "archiv": 3, "locat": 3, "nciglob": 3, "ncienterpris": 3, "gov": 3, "chang": [2, 3], "yara": 3, "abdou": 3, "medhavi": 3, "gupta": 3, "mariko": 3, "asaoka": 3, "kristoph": 3, "attwood": 3, "opyrch": 3, "mateusz": 3, "shipra": 3, "gandhi": 3, "kazuaki": 3, "takab": 3, "left": 3, "side": [2, 3], "associ": [2, 3], "biologi": 3, "wors": 3, "right": 3, "scientif": 3, "13377": 3, "august": 3, "com": 3, "articl": 3, "s41598": 3, "022": 3, "16749": 3, "doi": 3, "1038": 3, "katrina": 3, "korhonen": 3, "emili": 3, "conant": 3, "eric": 3, "cohen": 3, "mari": 3, "synnestvedt": 3, "elizabeth": 3, "mcdonald": 3, "susan": 3, "weinstein": 3, "simultan": 3, "acquir": 3, "versu": 3, "tomosynthesi": 3, "radiolog": 3, "292": 3, "juli": 3, "pub": 3, "1148": 3, "radiol": 3, "2019182027": 3, "lawrenc": 3, "w": 3, "bassett": 3, "karen": 3, "conner": 3, "iv": 3, "ms": 3, "holland": 3, "frei": 3, "medicin": 3, "6th": 3, "bc": 3, "decker": 3, "2003": 3, "ncbi": 3, "nlm": 3, "nih": 3, "nbk12642": 3, "rebecca": 3, "sawyer": 3, "lee": 3, "francisco": 3, "gimenez": 3, "assaf": 3, "hoogi": 3, "kana": 3, "kawai": 3, "miyak": 3, "mia": 3, "gorovoi": 3, "daniel": 3, "l": 3, "rubin": 3, "curat": 3, "aid": 3, "170177": 3, "decemb": 3, "2017": 3, "sdata2017177": 3, "sdata": 3, "david": 3, "bragg": 3, "philip": 3, "hedvig": 3, "hricak": 3, "oncolog": 3, "saunder": 3, "2nd": 3, "ed": 3, "2002": 3, "7216": 3, "7494": 3, "erin": 3, "princ": 3, "heidi": 3, "r": 3, "umphrei": 3, "mass": 3, "oxford": 3, "univers": 3, "press": 3, "march": 3, "academ": 3, "oup": 3, "24629": 3, "chapter": 3, "187959266": 3, "1093": 3, "med": 3, "9780190270261": 3, "0019": 3, "187965422": 3, "0026": 3, "elezabi": 3, "187962148": 3, "0023": 3, "grattan": 3, "guin": 3, "roger": 3, "cook": 3, "editor": 3, "landmark": 3, "write": 3, "western": 3, "1640": 3, "1940": 3, "elsevi": 3, "amsterdam": 3, "boston": 3, "1st": 3, "2005": 3, "9786610633739": 3, "medium": 3, "electron": [2, 3], "ebookcentr": 3, "proquest": 3, "lib": 3, "detail": 3, "action": 3, "docid": 3, "269620": 3, "stephen": 3, "m": 3, "stigler": 3, "invent": 3, "annal": 3, "1981": 3, "projecteuclid": 3, "1214": 3, "ao": 3, "1176345451": 3, "gonzalo": 3, "arc": 3, "nonlinear": 3, "wilei": 3, "intersci": 3, "hoboken": 3, "j": 3, "471": 3, "67624": 3, "klapper": 3, "harri": 3, "respons": 3, "gaussian": 3, "filter": 3, "ir": 3, "transact": 3, "au": 3, "1959": 3, "ieeexplor": 3, "ieee": 3, "1166198": 3, "1109": 3, "h": 3, "hwang": 3, "haddad": 3, "result": 3, "499": 3, "502": 3, "april": 3, "1995": 3, "370679": 3, "buad": 3, "coll": 3, "morel": 3, "denois": 3, "societi": 3, "confer": 3, "cvpr": 3, "san": 3, "diego": 3, "ca": 3, "usa": 3, "1467423": 3, "choos": [], "exampl": 2, "ex": [], "image_prep": [], "dive": 2, "somewhat": 2, "imprecis": 2, "sai": 2, "But": [], "imperfect": 2, "underli": 2, "optic": 2, "uneven": 2, "background": 2, "contribut": 2, "With": 2, "problem": 2, "state": 2, "abrupt": 2, "decomposit": 2, "surfac": 2, "charg": 2, "fluctuat": 2, "induc": 2, "thermal": 2, "environment": 2, "electromagnet": 2, "domin": 2, "rai": 2, "multiplicative_noise_model": [], "logarithm": 2, "becom": 2, "exponenti": 2, "log": 2, "additive_noise_model": [], "s_n": 2, "strategi": 2, "convert": 2, "appropri": 2, "lastli": 2, "consequ": 2, "error": 2, "equat": 2, "case": 2, "end": 2, "le": 2, "simplif": 2, "static": 2, "dynam": 2, "modifi": 2, "grai": 2, "black": 2, "255": 2, "dark": 2, "noise_typ": [], "gaussian_nois": [], "romain": 3, "lain": 3, "guillaum": 3, "jacquemet": 3, "alexand": 3, "krull": 3, "introduct": 3, "bioimag": 3, "era": 3, "intern": 3, "biochemistri": 3, "cell": 3, "140": 3, "106077": 3, "novemb": 3, "2021": 3, "linkinghub": 3, "retriev": 3, "pii": 3, "s1357272521001588": 3, "1016": 3, "biocel": 3, "alan": 3, "bovik": 3, "handbook": 3, "commun": 3, "multimedia": 3, "2000": 3, "119790": 3, "ve": 2, "implicitli": 2, "decompos": 2, "desir": 2, "uncorrupt": 2, "domain": 2, "grayscal": 2, "tabl": 2, "pepper": [], "speckl": [], "multivari": [], "mu": 2, "bigg": [], "symmetr": [], "matrix": [], "covari": [], "mathbb": [], "finit": [], "denot": [], "diagon": [], "diag": [], "2_1": [], "sigma_2": [], "whenev": 2, "chaotic": 2, "hand": 2, "order": 2, "magnitud": 2, "unsuspect": 2, "beauti": 2, "prove": 2, "latent": 2, "along": 2, "sir": 2, "franci": 2, "galton": 2, "1889": 2, "One": 2, "theorem": 2, "under": 2, "fairli": 2, "mild": 2, "hold": 2, "even": 2, "vibrat": 2, "atom": 2, "heat": 2, "circuit": 2, "sqrt": 2, "numref": [], "mmg_gaussian": [], "notic": 2, "fuzzi": 2, "anoth": 2, "ten": 2, "much": 2, "objection": 2, "correl": 2, "basi": 2, "triangl": 2, "q": 2, "uniform": 2, "graduat": 2, "lost": 2, "textur": 2, "constant": 2, "too": 2, "scallop": 2, "strike": 2, "reflect": 2, "becaus": 2, "microscop": 2, "rough": 2, "figur": [], "degrad": 2, "due": 2, "alter": 2, "unalt": 2, "easili": [], "lum": []}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"exploratori": 1, "data": 1, "analysi": 1, "eda": 1, "cbi": 1, "ddsm": 1, "dataset": 1, "The": 1, "case": 1, "dictionari": 1, "dicom": 1, "approach": 1, "guid": 1, "question": 1, "plan": 1, "univari": 1, "breast": 1, "densiti": 1, "left": 1, "right": 1, "side": 1, "imag": [1, 2], "view": 1, "abnorm": 1, "id": 1, "type": [1, 2], "subtleti": 1, "bi": 1, "rad": 1, "assess": 1, "calcif": 1, "distribut": 1, "mass": 1, "shape": 1, "margin": 1, "patholog": 1, "cancer": 1, "summari": 1, "bivari": 1, "target": 1, "variabl": 1, "associ": 1, "diagnosi": 1, "featur": 1, "multivari": 1, "build": 1, "pipelin": 1, "get": 1, "model": [1, 2], "execut": 1, "select": 1, "predict": 1, "result": 1, "import": 1, "metadata": 1, "photometr": 1, "interpret": 1, "sampl": 1, "per": 1, "pixel": 1, "bit": 1, "depth": 1, "height": 1, "width": 1, "size": 1, "aspect": 1, "ratio": 1, "valu": 1, "qualiti": 1, "benign": 1, "probabl": 1, "suspici": 1, "highli": 1, "preprocess": 2, "overview": 2, "denois": 2, "nois": 2, "gaussian": 2, "quantiz": 2, "poisson": 2, "filter": 2, "mean": 2, "refer": 3, "addit": 2, "multipl": 2, "impuls": 2, "screen": 2, "film": 2, "mammographi": 2, "speckl": 2, "salt": 2, "pepper": 2}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})
"""Mots-clés et détection floue sur la sortie HTR (graphies du XVIIIe s.)."""
import re, unicodedata
from rapidfuzz import fuzz

# (étiquette, radicaux normalisés). Radicaux courts = correspondance exacte de token seulement.
THEMES = [
    ("Juden", ["jud", "juden", "judenschaft", "judenschafft", "judengasse", "judenthor", "judenpforte", "judenkirchhof", "jüd", "jued", "jüdisch", "jüdische", "jüdischen", "jüdin", "juedisch"]),
    ("Rabbiner", ["rabbiner", "rabbi", "rabbinat", "rabiner", "rabbin", "rabbinen"]),
    ("Kiste", ["kiste", "kisten", "kasten", "küste", "coffre", "truhe"]),
    ("versiegelt", ["versiegelt", "versigelt", "versieglet", "siegel", "obsignirt"]),
    ("Schriften", ["schriften", "schrifften", "manuscript", "manuscripta", "hebräisch", "hebraisch", "hebreisch", "bücher", "bucher", "buecher"]),
    ("Feuer", ["verbrennen", "verbrant", "verbrannt", "feuer", "feur"]),
    ("Begraben", ["vergraben", "begraben", "begräbnis", "begrabnis", "kirchhof", "kirchhoff", "leiche", "leich"]),
    ("Bann", ["bann", "schulbann", "cherem", "herem", "excommunic"]),
    ("Sabbatai", ["sabbathai", "sabbatai", "sabbatianer", "sabbatianisch", "irrlehre", "ketzerei", "ketzer", "cabbala", "kabbala", "cabala"]),
    ("Censur", ["censur", "zensur", "büchercommission", "bücher-commission", "commissariat", "confiscirt", "confisciret", "confiscation"]),
    ("Nacht", ["nachts", "nacht", "bey nacht", "nächtlich"]),
    ("Thor", ["thor", "thore", "pforte", "aufschließen", "aufschliesen", "aufgeschlossen"]),
    ("Erlaubnis", ["erlaubnis", "erlaubnüs", "permission", "verstattet", "verstatten", "concession"]),
    ("Wacht", ["wacht", "schildwache", "schildwacht", "wache"]),
    ("Fuhre", ["fuhre", "fuhr", "wagen", "fuhrmann"]),
]

# Personnes liées à l'affaire. Prénoms seuls exclus (trop fréquents) : on cherche noms et composés.
# Source : P = dossier du projet (rapports v3, Sclar), W = vérifié en ligne (Ets Haim ms. Fuks 241, Kestenbaum), H = indiqué par Hillel, non vérifié.
PERSONS = [
    ("P:Luzzatto/Machal", ["luzzatto", "luzatto", "lucatto", "luzato", "lusatto", "luzzato", "lucato", "luzat", "lussatto",
                          "machal", "mahal", "mechal", "ramchal", "ramhal", "rmchl", "paduaner"]),
    ("P:Poppers", ["poppers", "popper", "poppert", "popers", "pobers", "boppers", "bopper", "pappers",
                  "jacob cohen", "jacob kohen", "jacob kahn", "jacob coen", "jakob cohen", "jacob von prag", "jacob prag", "oberrabbiner", "ober-rabbiner"]),
    ("P:Hagiz", ["hagiz", "hagis", "chagiz", "chagis", "chages", "hages", "haghiz", "hagies", "chagies"]),
    ("P:Bassan", ["bassan", "bassano", "bassani", "bassanus", "basan", "bassen", "jesaias bassan", "isaia bassan"]),
    ("P:Alpron", ["alpron", "alperon", "alprun", "alpronn", "halpron", "alparon", "alprone"]),
    ("P:Cracovia", ["cracovia", "cracau", "krakau", "krakow", "cracow", "krakauer", "cracauer", "kroke"]),
    ("P:Morpurgo", ["morpurgo", "morporgo", "morpurg", "marpurgo"]),
    ("P:Katzenellenbogen", ["katzenellenbogen", "katzenelnbogen", "katzenellnbogen", "katzenelenbogen", "ellenbogen"]),
    ("H:Nechemia Cohen", ["nechemia", "nehemia", "nehemias", "nechemias", "neemia", "nechemje", "nehemiah"]),
    ("W:Venise (signataires)", ["aboab", "abuab", "aboaf", "pacifico", "pacheco", "pachecco", "minz", "mintz", "altares", "altaras",
                                "merari", "paduani", "padovani", "belilios", "belillos", "belilius"]),
    ("W:Cercle de Padoue", ["gordon", "jekutiel", "kussiel", "kosiel", "valle", "della valle", "calvo", "vitale"]),
    ("W:Hayon (Hagiz, de loin)", ["hayon", "chajon", "chayon", "hajon", "chaion"]),
    ("Cohen (bruit fort)", ["cohen", "coen", "kohen", "cohn", "kohn", "kahn", "katz"]),
]
PLACES = [
    # Italie : villes de l'affaire (Padoue, Venise, Reggio, Ancône, Modène, Mantoue, Vérone, Livourne, Ferrare) et États
    ("Lieu:Italie", ["welsch", "welschen", "welscher", "welschland", "wälschland", "walschland", "italien", "italiaen", "italiänisch", "italiaenisch", "italia", "lombardey", "lombardei",
                     "padua", "padova", "paduaner", "padoue",
                     "venedig", "venetia", "venezia", "venedischer", "venetianisch", "venetianer", "venise", "republic venedig",
                     "reggio", "ancona", "ancone", "modena", "modenesisch", "mantua", "mantova", "mantuanisch",
                     "verona", "veronesisch", "livorno", "liuorno", "leghorn", "ferrara", "trieste", "triest", "kirchenstaat", "päpstlich"]),
    # Route des Alpes et Empire du Sud (Tyrol, Innsbruck, Trente, Bolzano, Augsbourg, Nuremberg, Bâle, Coire)
    ("Lieu:Route Alpes", ["tyrol", "tirol", "tyrolisch", "inspruck", "innspruck", "innsbruck", "insbruck", "trient", "trento", "botzen", "bozen",
                          "augspurg", "augsburg", "nürnberg", "nurnberg", "nürnberger", "basel", "chur", "graubündten", "graubunden", "bündten"]),
    # Nord : Altona, Hambourg, Wandsbek, Amsterdam, Provinces-Unies
    ("Lieu:Nord", ["altona", "altonaer", "hamburg", "hamburger", "hambourg", "wandsbeck", "wandsbek", "amsterdam", "amsterdamm", "holland", "holländisch", "generalstaaten", "niederlande"]),
    # Communautés ashkénazes qui ont condamné Ramḥal ou liées aux protagonistes (Sclar 2016) : Fürth, Berlin, Breslau, Brody, Lemberg, Nikolsburg, Prague, Vienne, Wilna
    ("Lieu:Ashkenaz", ["fürth", "furth", "fürther", "berlin", "breslau", "brody", "lemberg", "leopol", "nicolsburg", "nikolsburg", "prag", "praag", "böhmen", "wien", "wienn", "wilna", "vilna", "littauen", "pohlen", "polen"]),
]

# Paire « Moshe Ḥayyim » / « Moses Vita » : signalée seulement si les deux prénoms sont contigus, dans cet ordre.
# Vita = traduction italienne de Ḥayyim (« vie »).
PAIR_FIRST = ["moses", "moyses", "moises", "moise", "moisè", "mosè", "mosé", "mose", "moshe", "mosche", "mausche", "moische",
              "mojses", "moysis", "moyse", "mosses", "moscheh", "mosseh", "mos"]
PAIR_SECOND = ["chaim", "chajim", "chayim", "chayyim", "chaijm", "chaym", "chaiim", "haim", "hayim", "hayyim", "hajim", "haym",
               "chiam", "chaem", "vita", "vitta", "vitha", "vida"]
PAIR_LABEL = "P:Moshe Chayim / Moses Vita"

KEYWORDS = THEMES + PERSONS + PLACES

def norm(s: str) -> str:
    s = s.replace("ſ", "s").replace("ß", "ss").lower()
    s = s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 \-]", " ", s)

_PREP = [(lab, sorted({norm(t).strip() for t in terms})) for lab, terms in KEYWORDS]

_PF = sorted({norm(x).strip() for x in PAIR_FIRST}); _PS = sorted({norm(x).strip() for x in PAIR_SECOND})

def _tok_match(tok, variants, thr=86):
    best = 0
    for v in variants:
        if len(v) <= 4 or len(tok) <= 4:
            sc = 100 if tok == v else 0
        else:
            sc = int(fuzz.ratio(tok, v))
        if sc >= thr and sc > best:
            best = sc
    return best

def pair_hit(toks):
    """Moshe + Ḥayyim/Vita contigus (dans cet ordre)."""
    for a, b in zip(toks, toks[1:]):
        sa, sb = _tok_match(a, _PF), _tok_match(b, _PS)
        if sa and sb:
            return (PAIR_LABEL, f"{a} {b}", min(sa, sb))
    return None

def hits(text: str, thr: int = 86):
    """Renvoie [(étiquette, terme, score)] ; tokens courts (<=4) = égalité stricte."""
    t = norm(text)
    toks = [x for x in re.split(r"[\s\-]+", t) if x]
    out = []
    for lab, terms in _PREP:
        best = None
        for term in terms:
            if " " in term:
                sc = fuzz.partial_ratio(term, t)
                if sc < 95: sc = 0
            elif len(term) <= 4:
                sc = 100 if term in toks else 0
            else:
                sc = max((fuzz.ratio(term, tok[:len(term)+2]) if len(tok) >= len(term) - (1 if len(term) >= 7 else 0) else 0) for tok in toks) if toks else 0
                if sc < thr and any(term in tok for tok in toks):
                    sc = 95
            if sc >= thr and (best is None or sc > best[2]):
                best = (lab, term, int(sc))
        if best:
            out.append(best)
    ph = pair_hit(toks)
    if ph:
        out.append(ph)
    return out

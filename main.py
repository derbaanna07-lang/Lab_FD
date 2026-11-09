import sqlite3
from pathlib import Path
from sqlite3 import Error

import pandas as pd
import tokenize_uk as tok

# pymorphy2 не працює на Python 3.11+, тому пробуємо pymorphy3 (API той самий)
import pymorphy3 as pymorphy2      # pymorphy3 — сумісна заміна pymorphy2 для нових версій Python

from sklearn.feature_extraction.text import TfidfVectorizer

try:
    BASE = Path(__file__).parent            # запуск як python main.py
except NameError:
    BASE = Path.cwd()                       # Colab / Jupyter: поточна папка (/content)

TEXTS = {                                   # назва тексту -> файл
    "Sanatoriy": BASE / "Sanatoriy.txt",
    "FataMorgana": BASE / "FataMorgana.txt",
}

SAMPLE_SIZE = 20000                         # обсяг вибірки (слововживань), за методичкою
PART_SIZE = 1000                            # розмір підвибірки
N_PARTS = SAMPLE_SIZE // PART_SIZE          # 20 підвибірок

# для TF-IDF беремо лише повнозначні слова (інакше вгорі будуть «і», «не», «в»...)
CONTENT_ONLY = True
CONTENT_POS = ("NOUN", "ADJF", "ADJS", "VERB", "INFN", "PRTF", "PRTS", "GRND", "ADVB")

POS_UA = {
    "NOUN": "Іменник", "ADJF": "Прикметник", "ADJS": "Прикметник (коротка форма)",
    "VERB": "Дієслово", "INFN": "Інфінітив", "PRTF": "Дієприкметник",
    "PRTS": "Дієприкметник (коротка форма)", "GRND": "Дієприслівник",
    "NUMR": "Числівник", "ADVB": "Прислівник", "NPRO": "Займенник",
    "PRED": "Предикатив", "PREP": "Прийменник", "CONJ": "Сполучник",
    "PRCL": "Частка", "INTJ": "Вигук", "UNKN": "Невизначено",
}

PART_COLS = [f"part_{i}" for i in range(1, N_PARTS + 1)]
PART_DDL = ", ".join(f"{c} INTEGER" for c in PART_COLS)
PART_SUMS = ", ".join(f"SUM(part = {i})" for i in range(1, N_PARTS + 1))
TABLES = ("sentences", "words", "dict_of_freq", "lemma_freq", "pos_freq", "tfidf")


class Word:
    """Одне слововживання: слово, нормалізована словоформа, лема, частина мови."""
    s_morph = pymorphy2.MorphAnalyzer(lang='uk')
    _cache = {}

    def __init__(self, word, clean_word, lemma, pos):
        self._word = word
        self._clean_word = clean_word
        self._lemma = lemma
        self._pos = pos

    @staticmethod
    def from_token(token):
        clean = token.lower()
        if clean not in Word._cache:                       # кеш: кожну форму аналізуємо раз
            parsed = Word.s_morph.parse(clean)[0]
            Word._cache[clean] = (parsed.normal_form, parsed.tag.POS or "UNKN")
        lemma, pos = Word._cache[clean]
        return Word(token, clean, lemma, pos)

    def show(self):
        print(f"Слово : {self._word}   Словоформа : {self._clean_word}   "
              f"Лема : {self._lemma}   Частина мови : {self._pos}")

    def get_tuple(self):
        return (self._word, self._clean_word, self._lemma, self._pos)


class Sample:
    """Вибірка: перші SAMPLE_SIZE слововживань тексту, розбиті на речення."""

    def __init__(self, text, limit=SAMPLE_SIZE):
        self._sentences = []                # [(текст речення, [токени-слова])]
        total = 0
        for sent in tok.tokenize_sents(text):
            words = [w for w in tok.tokenize_words(sent) if w[0].isalpha()]
            if not words:
                continue
            words = words[:limit - total]
            self._sentences.append((" ".join(sent.split()), words))   # без переносів рядків
            total += len(words)
            if total >= limit:
                break
        self._size = total

    def size(self):
        return self._size                   # N — усі слововживання (з повторами)

    def get_sentences(self):
        return [(i + 1, s) for i, (s, _) in enumerate(self._sentences)]

    def get_words(self):
        """Рядки для таблиці words: (sent_id, part, word, clean_word, lemma, pos)."""
        rows, idx = [], 0
        for sent_id, (_, tokens) in enumerate(self._sentences, start=1):
            for token in tokens:
                part = idx // PART_SIZE + 1          # номер підвибірки 1..20
                rows.append((sent_id, part) + Word.from_token(token).get_tuple())
                idx += 1
        return rows


class SQL:
    def __init__(self, file):
        self._file = file
        self._connection = SQL.create_connection(file)
        self._cursor = self._connection.cursor()

    @staticmethod
    def create_connection(db_file):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
        except Error as e:
            print(e)

        return conn

    # ---------- створення та наповнення таблиць ----------
    def create_tables(self):
        for t in TABLES:
            self._cursor.execute(f"DROP TABLE IF EXISTS {t}")
        self._cursor.executescript('''
            CREATE TABLE sentences (sent_id INTEGER PRIMARY KEY, sentence TEXT);
            CREATE TABLE words (
                id INTEGER PRIMARY KEY, sent_id INTEGER, part INTEGER,
                word TEXT, clean_word TEXT, lemma TEXT, pos TEXT);
        ''')
        self._connection.commit()

    def fill_words(self, sample):
        self._cursor.executemany("INSERT INTO sentences VALUES (?, ?)", sample.get_sentences())
        self._cursor.executemany(
            "INSERT INTO words (sent_id, part, word, clean_word, lemma, pos) "
            "VALUES (?, ?, ?, ?, ?, ?)", sample.get_words())
        self._connection.commit()

    def build_freq_tables(self):
        """Усі частотні таблиці будуються агрегуючими запитами SELECT ... COUNT ... GROUP BY.
        COUNT(*) рахує КОЖНЕ слововживання, тому повтори слів враховуються."""
        c = self._cursor
        # БД_ЧС (частина мови + лема + словоформа)
        c.execute(f'''CREATE TABLE dict_of_freq (
            id INTEGER PRIMARY KEY, pos TEXT, lemma TEXT, wordform TEXT,
            abs_freq INTEGER, {PART_DDL})''')
        c.execute(f'''INSERT INTO dict_of_freq (pos, lemma, wordform, abs_freq, {", ".join(PART_COLS)})
            SELECT pos, lemma, clean_word, COUNT(*), {PART_SUMS}
            FROM words GROUP BY pos, lemma, clean_word
            ORDER BY COUNT(*) DESC, clean_word''')
        # БД_ЧС_лем
        c.execute(f'''CREATE TABLE lemma_freq (
            id INTEGER PRIMARY KEY, lemma TEXT, abs_freq INTEGER, {PART_DDL})''')
        c.execute(f'''INSERT INTO lemma_freq (lemma, abs_freq, {", ".join(PART_COLS)})
            SELECT lemma, COUNT(*), {PART_SUMS}
            FROM words GROUP BY lemma ORDER BY COUNT(*) DESC, lemma''')
        # БД_ЧС_ЧМ
        c.execute(f'''CREATE TABLE pos_freq (
            id INTEGER PRIMARY KEY, pos TEXT, pos_ua TEXT, freq INTEGER, {PART_DDL})''')
        c.execute(f'''INSERT INTO pos_freq (pos, freq, {", ".join(PART_COLS)})
            SELECT pos, COUNT(*), {PART_SUMS}
            FROM words GROUP BY pos ORDER BY COUNT(*) DESC''')
        c.executemany("UPDATE pos_freq SET pos_ua = ? WHERE pos = ?",
                      [(ua, pos) for pos, ua in POS_UA.items()])
        self._connection.commit()

    # ---------- вибірки / статистика ----------
    def select_pos_freq(self):
        print("\n--- Частини мови (SELECT pos, COUNT(*) ... GROUP BY pos) ---")
        res = self._cursor.execute(
            "SELECT pos, pos_ua, freq, ROUND(100.0 * freq / (SELECT SUM(freq) FROM pos_freq), 2) "
            "FROM pos_freq ORDER BY freq DESC")
        for pos, ua, freq, pct in res:
            print(f"{pos:6} {str(ua):32} {freq:7} {pct:6}%")

    def select_top_lemmas(self, n=15):
        res = self._cursor.execute(
            "SELECT lemma, abs_freq FROM lemma_freq ORDER BY abs_freq DESC LIMIT ?", (n,))
        print(f"\n--- Топ-{n} лем ---\n", res.fetchall())

    def select_other_stats(self):
        c = self._cursor
        n, forms, lemmas = c.execute(
            "SELECT SUM(abs_freq), COUNT(*), (SELECT COUNT(*) FROM lemma_freq) "
            "FROM dict_of_freq").fetchone()
        hapax = c.execute("SELECT COUNT(*) FROM dict_of_freq WHERE abs_freq = 1").fetchone()[0]
        avg_len = c.execute("SELECT AVG(LENGTH(clean_word)) FROM words").fetchone()[0]
        sents = c.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        print("\n--- Інші характеристики ---")
        print(f"N (слововживань): {n}")
        print(f"різних словоформ: {forms}, різних лем: {lemmas}")
        print(f"TTR (словоформ/N): {forms / n:.3f}")
        print(f"гапаксів (слів із частотою 1): {hapax}")
        print(f"середня довжина слова: {avg_len:.2f} літер")
        print(f"речень: {sents}, середня довжина речення: {n / sents:.2f} слів")

    def show_table(self, name, n=8):
        df = pd.read_sql_query(f"SELECT * FROM {name} LIMIT {n}", self._connection)
        print(f"\n=== Таблиця {name} (перші {n} рядків) ===")
        print(df.to_string(index=False))

    def lemma_counts(self):
        """Лема -> частота (для TF-IDF)."""
        q = "SELECT lemma, COUNT(*) FROM words "
        if CONTENT_ONLY:
            q += "WHERE pos IN (%s) " % ",".join("'%s'" % p for p in CONTENT_POS)
        q += "GROUP BY lemma"
        return self._cursor.execute(q).fetchall()

    def save_tfidf(self, rows):
        self._cursor.execute("CREATE TABLE tfidf (id INTEGER PRIMARY KEY, lemma TEXT, tfidf REAL)")
        self._cursor.executemany("INSERT INTO tfidf (lemma, tfidf) VALUES (?, ?)", rows)
        self._connection.commit()

    def export_xlsx(self, path):
        try:
            with pd.ExcelWriter(path) as xw:
                for t in TABLES:
                    pd.read_sql_query(f"SELECT * FROM {t}", self._connection)\
                        .to_excel(xw, sheet_name=t, index=False)
            print("Збережено:", path)
        except Exception as e:                     # немає openpyxl — зберігаємо CSV
            print("Excel не вдалося (", e, ") — зберігаю CSV")
            for t in TABLES:
                pd.read_sql_query(f"SELECT * FROM {t}", self._connection)\
                    .to_csv(f"{path}.{t}.csv", index=False, encoding="utf-8-sig")


def compute_tfidf(dbs, top=15):
    """TF-IDF по лемах (sklearn, як у прикладі викладача). Документ = усі слововживання лем тексту."""
    names = list(dbs)
    docs = [" ".join((lemma + " ") * f for lemma, f in dbs[n].lemma_counts()) for n in names]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    tfidf_matrix = vectorizer.fit_transform(docs)
    features_names = vectorizer.get_feature_names_out()
    for i, name in enumerate(names):
        scores = tfidf_matrix.toarray()[i]
        sorted_keywords = [(word, float(score)) for score, word in
                           sorted(zip(scores, features_names), reverse=True) if score > 0]
        dbs[name].save_tfidf(sorted_keywords)
        print(f"\n--- TF-IDF, {name}: топ-{top} ---")
        for word, score in sorted_keywords[:top]:
            print(f"{word:20} {score:.4f}")


if __name__ == "__main__":
    dbs = {}
    for name, path in TEXTS.items():
        with open(path, encoding='utf-8-sig', mode='r') as f:   # utf-8-sig прибирає BOM
            text = f.read()
        sample = Sample(text)
        print(f"\n########## {name}: у вибірці {sample.size()} слововживань "
              f"({N_PARTS} підвибірок по {PART_SIZE})")

        sql = SQL(str(BASE / f"freq_{name}.db"))
        sql.create_tables()
        sql.fill_words(sample)
        sql.build_freq_tables()
        dbs[name] = sql

    compute_tfidf(dbs)

    for name, sql in dbs.items():
        print(f"\n########## {name}")
        for t in TABLES:
            sql.show_table(t)
        sql.select_pos_freq()
        sql.select_top_lemmas()
        sql.select_other_stats()
        sql.export_xlsx(str(BASE / f"tables_{name}.xlsx"))

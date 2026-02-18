import pandas as pd
import re
import logging
from typing import Any
from sklearn.preprocessing import LabelEncoder
from base import ProcessingStep

logger = logging.getLogger(__name__)

class DevFilter(ProcessingStep):
    """Оставляет только записи, относящиеся к IT-разработке."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрует DataFrame по расширенному списку ключевых слов."""
        logger.info("Фильтрация IT-специалистов...")
        keywords = [
            'разработчик', 'developer', 'programmer', 'программист',
            'frontend', 'backend', 'fullstack', 'python', 'java', 'c++', 'php'
        ]

        mask = df.apply(
            lambda x: x.astype(str).str.lower().str.contains('|'.join(keywords)).any(),
            axis=1
        )

        df_dev = df[mask].copy()
        logger.info(f"Осталось записей после фильтрации: {len(df_dev)}")
        return super().process(df_dev)

class LevelLabeler(ProcessingStep):
    """Размечает уровень (Junior/Middle/Senior) на основе текста и опыта."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Проводит разметку уровней для каждой строки."""
        logger.info("Разметка уровней (Junior/Middle/Senior)...")

        def get_level(row: pd.Series) -> str:
            text = " ".join(row.astype(str)).lower()
            senior_keys = ['senior', 'саньор', 'сеньор', 'ведущий', 'lead', 'лид', 'architect']
            junior_keys = ['junior', 'джуниор', 'младший', 'intern', 'стажер', 'trainee']

            if any(x in text for x in senior_keys): return 'Senior'
            if any(x in text for x in junior_keys): return 'Junior'
            if any(x in text for x in ['middle', 'мидл']): return 'Middle'

            exp_col = next((col for col in df.columns if 'опыт' in col.lower()), None)
            if exp_col and 'без опыта' in str(row[exp_col]).lower():
                return 'Junior'
            return 'Middle'

        df['target_level'] = df.apply(get_level, axis=1)
        df = df[df['target_level'].isin(['Junior', 'Middle', 'Senior'])]
        return super().process(df)

class ClassificationFeatureEncoder(ProcessingStep):
    """Извлекает числовой опыт работы и кодирует признаки."""

    def _extract_experience_months(self, text: str) -> int:
        """Переводит строку опыта в число месяцев через регулярные выражения."""
        if not isinstance(text, str): return 0
        years = re.findall(r'(\d+)\s*(?:год|лет|г\.)', text.lower())
        months = re.findall(r'(\d+)\s*(?:месяц|мес\.)', text.lower())
        return sum(int(y) * 12 for y in years) + sum(int(m) for m in months)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Трансформирует текстовые признаки в числовой формат."""
        logger.info("Глубокое извлечение признаков (Опыт + ЗП)...")
        
        exp_col = next((col for col in df.columns if 'опыт' in col.lower()), None)
        if exp_col:
            df['experience_months'] = df[exp_col].astype(str).apply(self._extract_experience_months)
            df.loc[df[exp_col].astype(str).str.contains('без опыта', case=False), 'experience_months'] = 0
            df = df.drop(columns=[exp_col])

        target_col = next((col for col in df.columns if 'ЗП' in col.upper()), None)
        if target_col:
            raw_salary = df[target_col].astype(str).str.replace(r'[^\d]', '', regex=True)
            df['salary_feature'] = pd.to_numeric(raw_salary, errors='coerce').fillna(0).astype(float)
            df = df.drop(columns=[target_col])

        for col in [c for c in df.columns if c != 'target_level']:
            if df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        return super().process(df)

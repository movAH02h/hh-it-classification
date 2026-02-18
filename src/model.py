import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from base import ProcessingStep

logger = logging.getLogger(__name__)

class ClassificationTrainer(ProcessingStep):
    """Обучает RandomForest на числовых признаках."""

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выполняет обучение и визуализацию матрицы ошибок."""
        logger.info("Обучение RandomForest на числовых признаках...")
        X = df.drop(columns=['target_level'])
        y = df['target_level']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight={'Junior': 12.0, 'Middle': 1.0, 'Senior': 4.0},
            random_state=42,
            n_jobs=-1
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        logger.info(f"\nОТЧЕТ:\n{classification_report(y_test, y_pred)}")

        plt.figure(figsize=(7, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges',
                    xticklabels=clf.classes_, yticklabels=clf.classes_)
        plt.title('Матрица ошибок')
        plt.show()

        return df

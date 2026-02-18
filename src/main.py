import os
import logging
import pandas as pd
from utils import get_user_file_path
from loaders import FileLoader
from cleaners import MojibakeCorrector
from processing import DevFilter, LevelLabeler, ClassificationFeatureEncoder
from model import ClassificationTrainer

logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def run_classification_app() -> None:
    """
    Запускает PoC (Proof of Concept) классификации вакансий.
    Реализует логику финальной ячейки твоего Google Colab.

    Вернет:
        None
    """
    csv_path = get_user_file_path(default_filename="hh_cleaned.csv")
    
    if not csv_path:
        logger.error("Работа программы прервана: файл не выбран.")
        return

    logger.info(f"Запуск конвейера классификации для файла: {csv_path}")

    pipeline = FileLoader()
    
    (pipeline.set_next(MojibakeCorrector())
             .set_next(DevFilter())
             .set_next(LevelLabeler())
             .set_next(ClassificationFeatureEncoder())
             .set_next(ClassificationTrainer()))

    try:
        pipeline.process(csv_path)
        logger.info("Классификация завершена успешно.")
        
    except FileNotFoundError as e:
        logger.error(f"Ошибка: Файл не найден — {e}")
    except Exception as e:
        logger.error(f"Критическая ошибка пайплайна: {e}")
        import traceback
        logger.debug(traceback.format_exc())

if __name__ == "__main__":
    run_classification_app()

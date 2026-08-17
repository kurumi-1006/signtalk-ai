from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    device_id: str
    device_key_id: str
    device_secret: str
    api_url: str
    outbox_db_path: str = './data/outbox.db'
    log_level: str = 'INFO'
    mock_mode: bool = True
    model_path: str = './models/vsl_metric_lowshot/best_vsl_metric_encoder.pt'
    labels_path: str = './models/vsl_metric_lowshot/labels.csv'
    active_model_id: str = 'vsl_metric_lowshot'
    edge_host: str = '0.0.0.0'
    edge_port: int = 8081
    cors_origins: str = '*'
    min_word_confidence: float = 0.62
    min_word_margin: float = 0.10
    min_landmark_coverage: float = 0.5

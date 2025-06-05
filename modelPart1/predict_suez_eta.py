import psycopg2
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from modelPart1.pg_dataset_eta import PgETADataset
from modelPart1.eta_eta_predictor import ETAPredictorNet
from modelPart1.eta_speed_model import GroupEmbedder, NewsEmbedder
from modelPart1.utils import eval_eta, collate_fn_eta


DB_DSN = "dbname=eta_voyage2 user=cxsj host=localhost port=5433"
MODEL_PATH = Path("output/best.pth")


def query_suez_voyage_ids(dsn: str) -> list[int]:
    """Query voyage IDs of routes that pass through the Suez Canal."""
    sql = """
        SELECT DISTINCT voyage_id
        FROM voyage_node
        WHERE latitude BETWEEN 29.8 AND 31.3
          AND longitude BETWEEN 32.2 AND 32.6
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            return [r[0] for r in rows]
    finally:
        conn.close()


def build_dataloader(voy_ids: list[int], batch_size: int = 1):
    """Construct DataLoader limited to the given voyage IDs."""
    dataset = PgETADataset(train=False, k_near=32, h_ship=10, radius_km=50.0, step=32)
    dataset.voy_ids = voy_ids
    collate = lambda b: collate_fn_eta(b, H=dataset.H, K=dataset.K)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate)


def load_model(device: torch.device):
    Aemb = GroupEmbedder().to(device)
    shipemb = GroupEmbedder().to(device)
    nearemb = GroupEmbedder().to(device)
    mdl = ETAPredictorNet().to(device)

    if MODEL_PATH.exists():
        ckpt = torch.load(str(MODEL_PATH), map_location=device)
        Aemb.load_state_dict(ckpt['Aemb'])
        shipemb.load_state_dict(ckpt['shipemb'])
        nearemb.load_state_dict(ckpt['nearemb'])
        mdl.load_state_dict(ckpt['model'])
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    return mdl, Aemb, shipemb, nearemb


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voy_ids = query_suez_voyage_ids(DB_DSN)
    val_loader = build_dataloader(voy_ids)
    mdl, Aemb, shipemb, nearemb = load_model(device)
    mare, mae, rmse, mare_ci, mae_ci, rmse_ci = eval_eta(
        mdl,
        Aemb,
        shipemb,
        nearemb,
        val_loader,
        device,
        criterion=torch.nn.L1Loss(),
        use_amp=False,
    )
    print(
        "Suez voyages MARE: {:.3f} [{:.3f}, {:.3f}], MAE: {:.3f} [{:.3f}, {:.3f}], RMSE: {:.3f} [{:.3f}, {:.3f}]".format(
            mare,
            mare_ci[0],
            mare_ci[1],
            mae,
            mae_ci[0],
            mae_ci[1],
            rmse,
            rmse_ci[0],
            rmse_ci[1],
        )
    )


if __name__ == "__main__":
    main()

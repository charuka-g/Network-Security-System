<<<<<<< HEAD
=======
### Phishing URL Detection (MLOps Project)

(./architecture.png)

This repository implements an end‑to‑end **Network Security / Phishing Detection** system using modern **MLOps** practices:

- **Data ingestion** from MongoDB
- **Data validation** and **transformation**
- **Model training** and tracking with **MLflow / DagsHub**
- **Model packaging & serving** via a **FastAPI** web service
- **CI/CD & deployment** to AWS using GitHub Actions and Docker

---

### Project Structure (High‑Level)

- `app.py` – FastAPI application exposing:
  - `GET /train` – run the full training pipeline
  - `POST /predict` – upload CSV and get predictions rendered as an HTML table
- `main.py` – script entrypoint that runs the full training pipeline from the CLI.
- `push_data.py` – utility to push CSV network/phishing data into MongoDB.
- `networksecurity/` – Python package containing:
  - `components/` – pipeline components (`data_ingestion`, `data_validation`, `data_transformation`, `model_trainer`, etc.)
  - `entity/` – configuration and artifact entity definitions
  - `exception/` – custom exception handling (`NetworkSecurityException`)
  - `logging/` – project‑wide logging configuration
  - `utils/` – helper utilities (ML utils, IO helpers, etc.)
- `.github/workflows/main.yml` – GitHub Actions workflow for CI/CD to AWS.

---

### Environment Variables

Create a `.env` file in the project root (same folder as `app.py` and `main.py`). At minimum you should define:

```env
# MongoDB connection strings
MONGODB_URL_KEY="mongodb+srv://<user>:<password>@<host>/<db>?retryWrites=true&w=majority"
MONGO_DB_URL="mongodb+srv://<user>:<password>@<host>/<db>?retryWrites=true&w=majority"

# (Recommended) MLflow / DagsHub tracking
MLFLOW_TRACKING_URI="https://dagshub.com/<user>/<repo>.mlflow"
MLFLOW_TRACKING_USERNAME="<dagshub-username-or-token-id>"
MLFLOW_TRACKING_PASSWORD="<dagshub-token-or-password>"
```

These variables are loaded using `python-dotenv` via `load_dotenv()` in the relevant modules.

---

### Local Setup & Usage

1. **Clone and create environment**

   ```bash
   git clone <this-repo-url>
   cd NetworkSecurity

   # Recommended: create a virtualenv / conda env
   conda create -n networksecurityenv python=3.11 -y
   conda activate networksecurityenv

   pip install -r requirements.txt
   ```

2. **Configure `.env`**

   - Fill in MongoDB URI(s) and, optionally, MLflow credentials as shown above.

3. **Push initial data to MongoDB (optional)**

   ```bash
   python push_data.py
   ```

4. **Run the training pipeline from CLI**

   ```bash
   python main.py
   ```

5. **Run the FastAPI service**

   ```bash
   python app.py
   ```

   The API will start on `http://0.0.0.0:8000` by default. Visit:

   - `http://localhost:8000/docs` – interactive Swagger UI.
   - `GET /train` – trigger a training run.
   - `POST /predict` – upload a CSV file and get phishing / network predictions.

---

### GitHub Secrets (for CI/CD to AWS)

In your GitHub repository settings, configure the following **Secrets** used by `.github/workflows/main.yml`:

- `AWS_ACCESS_KEY_ID` – IAM access key ID with permissions for ECR, ECS/S3, etc.
- `AWS_SECRET_ACCESS_KEY` – secret key for the above IAM user.
- `AWS_REGION` – AWS region (e.g. `us-east-1`).
- `AWS_ECR_LOGIN_URI` – ECR login URI, e.g. `788614365622.dkr.ecr.us-east-1.amazonaws.com/networkssecurity`.
- `ECR_REPOSITORY_NAME` – ECR repository name, e.g. `networkssecurity`.

These are used by the workflow to:

- Authenticate to AWS
- Build and push the Docker image to ECR
- Run the container in your target environment

---

### Docker Setup on EC2 (Manual)

On a new Ubuntu EC2 instance, install and enable Docker:

```bash
# Optional system update
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Allow current user to run docker without sudo (example: ubuntu)
sudo usermod -aG docker ubuntu
newgrp docker
```

After Docker is installed and GitHub Actions has pushed an image to ECR, you can run the container (simplified example):

```bash
docker run -d -p 8080:8080 --ipc="host" --name=networksecurity \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION=$AWS_REGION \
  <AWS_ECR_LOGIN_URI>/<ECR_REPOSITORY_NAME>:latest
```

---

### Notes

- Sensitive values like database passwords, MLflow tokens, and AWS keys **must never be committed** to Git. Use `.env` locally and **GitHub Secrets** in CI.
- For production, consider using a secrets manager (AWS Secrets Manager, Parameter Store, etc.) instead of plain `.env` files.
>>>>>>> master

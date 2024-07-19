## Problem Description: Prediction Model for 0-DTE Trading


Same-day expiration options trading, also known as 0-DTE (zero days to expiration) trading, is a high-stakes, fast-paced strategy where traders buy and sell options contracts that expire on the same day they're purchased. This approach allows traders to capitalize on short-term price movements and potentially earn significant profits in a matter of hours.

### Challenge
This is a data and modelling problem. Options traders face the challenge of making quick, informed decisions on whether to purchase call or put options with same-day expiration for AAPL stock. The rapidly changing market conditions and the short time frame make it difficult to predict the closing price accurately.

### Solution
This project addresses the problem by implementing a principled prediction model that forecasts the closing price of a given stock. The model is designed specifically for daily options trading, providing traders with a tool to make informed decisions on 0-day expiration options.

### Key Features of the Solution
- Daily model retraining to incorporate the most recent market data
- Weekly model re-optimization using Optuna for hyperparameter optimization
- Continuous monitoring of data drift and target drift 
- Mechanism to trigger full re-optimization when necessary
Use a prediction model to forecast the closing price of AAPL and make informed decisions to purchase call or put options with a 0-day expiration (same-day expiration).

### Mock example
Scenario: Apple Inc. (AAPL) stock is priced at $150 when the market opens. Our trader uses the prediction model described in the project to forecast AAPL's closing price. The model suggests AAPL will end the day above $152.

Based on this prediction, the trader decides to buy 10 "call options" for AAPL. These options give the trader the right to buy AAPL shares at $151.90 (called the "strike price") by the end of the day. Each option costs $0.50, so the trader spends $500 in total (10 options x 100 shares per option x $0.50).

- **Winning scenario**: If AAPL indeed closes at $153, the trader was right. They can now buy AAPL shares at $151 (their agreed price) and immediately sell them at the market price of $153, making $2 per share. With 1000 shares (10 options x 100 shares each), they make $2000. Subtracting their initial $500 investment, their profit is $1500.
- **Losing scenario**: However, if the prediction is wrong and AAPL closes at $149, the options become worthless. The trader wouldn't want to buy shares at $151 when they can get them cheaper in the market. In this case, the trader loses their entire $500 investment.

This example illustrates the high-risk, high-reward nature of same-day options trading. The trader's decision to buy options is based on their prediction model, which is a useful tool to make informed decisions.

> For more details on the modelling strategies and implementation, see `backend > readme_backend.md`

## Cloud Architecture

0-DTE Trading is a sensitive and confidential business. For security, this MLOps system employs a cloud-agnostic design, prioritizing security and control while maintaining flexibility for potential cloud deployment. 

The architecture leverages cloud-native technologies but is primarily configured for on-premises or self-hosted environments to safeguard sensitive trading operations.

### Key architectural decisions
![image](assets/system_architecture.png)

1. Containerization: Docker Compose orchestrates our services, ensuring portability across environments, be it on-premises or cloud-based.

2. Reverse Proxy: Caddy handles HTTPS termination and acts as a reverse proxy, providing a uniform interface regardless of underlying infrastructure. For production, additional security measures such as proper certificate management and secrets handling must be implemented.

3. Data Storage:
   - PostgreSQL for transactional data, deployable both on-premises and as a managed cloud service if needed.
   - MinIO for object storage, offering S3-compatible interfaces that allow seamless transition to cloud object storage if required.

4. Experiment Tracking: Centralized MLflow instance, deployable on-premises or in the cloud, ensuring consistent experiment management across environments.

5. Workflow Orchestration: Prefect manages task scheduling with the flexibility to scale on-premises or in cloud environments.

6. Domain Model: Persistence is currently in-memory and via MLFlow. For more robust fail-safe solutions, It can be transition to a scalable solution like TimescaleDB, which supports both on-premises and cloud deployments.

### Scalability Considerations
- Worker processes can scale horizontally, either on high-compute servers on-premises or leveraging cloud-based Kubernetes clusters.
- Database and storage solutions chosen for their ability to scale in both on-premises and cloud scenarios. For cloud deployments, storage can be decoupled from the docker network and accessed via secure endpoints, allowing integration with highly-scalable managed services. Examples include Amazon FinSpace for financial analytics data or Google Cloud Bigtable for high-volume time-series data, both offering the performance and compliance features crucial for financial applications.

### Security and Deployment:
- The current setup prioritizes on-premises deployment for maximum data security.
- Cloud deployment is possible with additional security measures, including encryption, https via caddy, secrets management, network isolation, and cloud-specific security best practices.

This architecture balances security, scalability, and flexibility. While optimized for on-premises use, its cloud-agnostic design allows for potential cloud migration or hybrid setups in the future, adapting to evolving business needs while maintaining control over sensitive operations.

## Experiment Tracking & Model Registry
Our MLOps cycle leverages MLFlow for experiment tracking and model registry, integrated with Optuna for hyperparameter optimization. This setup ensures robust model management, reproducibility, and continuous improvement.

### Key Components

1. Experiment Tracking
    - Custom MLFlow wrapper functions (see `backend/src/model.py`)
    - Optuna-driven hyperparameter tuning for:
        - Number of lagged days (range: 2 to 5)
        - Feature selection using Recursive Feature Elimination (RFE)
    - MLFlow logs all parameters, metrics, and artifacts for each experiment

2. Optimization Process
    - Rolling window approach: 30 days historical, 10 days prediction
    - Multi-objective optimization balancing RMSE and directional accuracy
    - Optuna trials logged as separate MLFlow runs for detailed analysis
    - Pareto front selection for final model parameters

3. Model Registry
    - MLFlow maintains versioned history of all models
    - Models progress through stages: Development, Staging, Production
    - Full model lineage tracked, including training data and parameters

4. Daily Retraining
    - Automated daily model updates using most recent 10 trading days
    - New MLFlow run created for each retraining, logging updated model and metrics
    - Seamless model version management in registry

5. Monitoring and Re-optimization
    - Continuous tracking of prediction error (RMSE) and directional accuracy
    - Performance metrics logged daily in MLFlow
    - Automated re-optimization triggers if:
        - 3-day moving average of RMSE increases by 20%, or
        - Directional accuracy drops below 55%

To summarize, MLFlow is leverged for the following functions in the MLOps Cycle:
- Parameter Logging: Records all hyperparameters for each experiment
- Metric Tracking: Logs key performance metrics for easy comparison
- Artifact Storage: Stores feature importance plots, model checkpoints, and validation results
- Run Tagging: Tags each experiment run with relevant metadata
- Version Control: Maintains versioned history of all models
- Model Staging: Manages model progression through development stages
- Model Lineage: Tracks full lineage of each model

This MLOps cycle, powered by MLFlow and Optuna, helps to ensure consistent model performance, facilitates experimentation, and maintains an up-to-date model registry. The use of wrapper functions abstracts the complexity for day to day dev work while providing traceability, reproducibility, and version control for the 0-DTE trading system.

## Workflow Orchaestration

Our 0-DTE trading system leverages Prefect for workflow orchestration.

The workflow deployment utilizes client-side process workers, as implemented in the PrefectConfig class (`backend/src/orchaestrate.py`). This approach allows for flexible, on-premises execution of workflows, aligning with our security requirements. The PrefectConfig class sets up the necessary Prefect infrastructure, including a MinIO storage block for flow storage and a dedicated work pool for task execution.

### Key features of Prefect implementation

- Flow Deployment: Workflows are defined as Prefect flows and deployed programmatically. The deploy_flow function (e.g. in test_prefect_client.py) handles flow publication to storage, deployment creation, and optional immediate execution.
- Storage Management: Flows are stored in MinIO, providing a secure, S3-compatible storage solution that keeps our sensitive trading algorithms on-premises. 
    - !! note: currently approach uses `storage blocks`, which are deprecated. They need to be refactored to use `prefect_aws`. TBD. 
- Work Pool Configuration: A dedicated work pool is created for our 0-DTE trading tasks, ensuring isolated execution of our workflows.
- Scheduled Execution: Daily retraining workflows are scheduled using Prefect's built-in scheduling capabilities, ensuring consistent model updates. This is done via the UI.
- Dynamic Task Creation: Our implementation allows for dynamic creation and execution of tasks based on current trading conditions and model performance, via monitoring.

This comprehensive workflow orchestration setup ensures that our daily retraining, monitoring, and re-optimization processes run smoothly and reliably. The use of Prefect provides us with clear visibility into task execution and simplifies error handling. It centralizes and makes visible the complexities of a MLOps pipeline.

## Model Deployment
Covered above. 

## Model Monitoring
Covered above. 

## Reproducibility

There are several steps to this. 

Assumptions
- System - You are executing on a linux-environment. (I did not test or account for Windows)
- Docker - You are familiar with docker compose, basics of HTTP and reverse proxy
- UI - You have some experience with object storage and Prefect as there are parts of the config which is done in UI.
- Dev Environment - You are familiar with Poetry and PyEnv. The system is built on Python 3.8 because of work considerations (we use python 3.8 inhouse.)

Key Points
- Makefiles - yes, this is implemented extensively
- Tests - partially implemented. Written as vanilla functions, yet to be rewritten for pytest
- CICD - none implmented via GH ACtions or equivalent 
- Terraform - not implemented, but simplified through the use of docker-compose, makefiles and entrypoint shell scripts 


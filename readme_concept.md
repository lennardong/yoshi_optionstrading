# Software and Solution Architecture 
# Brief:
This is a document seeking investment for our product. It will be reviewed by seed-stage investors to judge the quality of our product. 

It should be confident, technical-but-exec-friendly 
It should take the tone and style of a seasoned developer. 
The language should follow software engineering lingo. 

Requsts from the VCs:
- [ ]  Software architecture diagrams – showing key components and their interaction with each other (class diagrams, interaction diagrams, etc.)
- [ ]  Tech stack diagram / core programming language(s)
- [ ]  Advanced technologies? Eg AI/ML, Blockchain etc.

--- 

# System Architecture V2 

The product architecture is built on Domain-Driven Design (DDD) principles, ensuring a clear separation of concerns and a scalable, maintainable codebase. By decomposing our product goals into logical modules, we've created a flexible and extensible system that can adapt to evolving business requirements.

This document will walk you through our modular architecture, detailing the design patterns and inter-module interactions that form the backbone of our innovative solution.

## Domain Layer

### The Core Domain Model (Atlas)
(quote on domain model)
The core of our system is the Atlas module, a OOP-heavy model built on  DDD concepts:
- Value Objects: Immutable objects representing descriptive aspects of the domain, such as unit management, chemical compounds, and reaction sets.
- Entities: Objects with distinct identities that run through time and different states, such as equipment and streams.
- Collections: Specialized containers for managing groups of related objects, including parameters and properties.
- Aggregate Roots: Clusters of domain objects treated as a single unit, with plants serving as our primary aggregate.

Where possible, each object is built on available industry schemas. For example, our compound library currently implements the CAS (Chemical Abstracts Service) schema, providing a standardized and widely recognized system for chemical substance identification.

Atlas utilizes a graph-based data structure, offering several key advantages:
- Technology agnosticism, allowing for flexible integration with various tech stacks.
- Intuitive modeling of complex domain problems.
- Efficient traversal and manipulation of interconnected data.
- Scalability of relationships. For example,  we utilize a straightforward graph-of-graphs approach to model facility and portfolio-level interactions for multi-plant contracts. 

We've invested significant resources in developing a comprehensive class hierarchy for equipment and stream types. This approach enables:
- High modularity and reusability across diverse implementations.
- Easy extension to accommodate new equipment or stream types.
- Consistent interface for interacting with different components.

We observe that the adoption of a domain-specific abstraction model has accelerated the development process and enabled powerful collaborations between process engineers and the software engineers.

### Process Simulators & AI Models
(qyote on modelling)

Our domain model forms the backbone of our physics-based and AI-driven modeling modules. By leveraging a rich, semantically consistent core representation, we've enabled seamless integration between process simulations and AI models. This approach has resulted in a significant reduction in development time and the ability to confidently ensemble multiple models, increasing value for users. 

Matrix is a integration module implementing a ports-and-adapters pattern to interface with external process simulators like gProms, AspenPlus, and HYSYS. This design ensures loose coupling between our core system and external simulators and standardized internal API for interacting with diverse simulation tools.

Daedalus is our abstraction layer with AI frameworks.W e built custom wrappers and abstract classes around cutting-edge frameworks like TorchRL, PyTorch, and LightningAI. This abstraction layer insulates our core functionality from the rapid changes in the AI landscape. As new AI technologies emerge or current ones evolve, we are able to swap out or upgrade the underlying libraries without disrupting the existing codebase.

Both modules are intentionally built to share similar interfaces, a design decision to facilitate seamless integration with our optimization module. This architectural choice enables a unified approach to physics-based and AI-driven modeling.

We envision this "hot-swappable, model-interoperability" to have compounding benefits effects as we add more and more modelling approaches to these modules.
  
### Parameter Optimization & Model Management (Metis)

The Metis module serves as our optimization engine, combining hyperparameter tuning, experiment tracking, and model management into a unified abstraction layer. 

By consolidating essential MLOps functions into a single abstraction, Metis not only reduces time-to-value for parameter optimization but also ensures consistency, reproducibility, and compliance, which are critical factors for enterprise adoption.

Metis is built on our core domain model and uses Dependency Injection to integrate with Matrix and Daedelus modules via a shared interface. This design decision has simplified engineering and allowed our engineers to focus on model innovation rather than infrastructure management. 

Under the hood, Metis is built upon various best-in-class technologies: 
- For Gradient-based optimization, we are leveraging Optuna for its parallelization and distributed computation capabilities.
- For Gradient-free optimization, we utilize our in-house libraries, based on research by Dr. Sushant Garud, providing a unique edge in handling complex, non-differentiable optimization problems.
- Finally, for Experiment tracking and model management, weintegrated with MLflow, an industry-standard open-source solution, ensuring reproducibility and facilitating collaboration between process engineers. 

In the implementation of these technologies, we've taken an opinionated approach and tuned them to the specific domain problems of chemical process plants. Based on (give project example and X rate increase in performance. give chart)

### Monitoring & Analytics (Athena)

The Athena module is our stateless analytics module, designed for high-performance computation across a wide range of critical metrics:

Some examples are: 
- Model drift detection: Ensuring the ongoing accuracy and reliability of our predictive models.
- Economic analysis: Providing real-time insights into the financial implications of process optimizations.
- Performance benchmarking: Continuously evaluating and improving system efficiency.

The breadth of this module means that we use a host of technologies such as scikit-learn, evidently and other proven algorithms. 

Unlike the previous modules, athena is decoupled from our core domain model.  The decoupling of Athena from our core domain model offers some key advantages:

Performance: As a stateless module, Athena leverages functional programming paradigms and vector operations, enabling high-throughput data processing without the overhead of object-oriented structures. This design choice facilitates efficient scaling for big data workloads, critical for real-time analytics in industrial settings.

Developer Efficiency: The loose coupling enables independent evolution of analytics capabilities, allowing data scientists to develop and deploy new value-added model insights without navigating complex domain-specific implementations. This separation of concerns has accelerated our development cycles, reduced time-to-market for new analytics features and significantly eased onboarding for new team members.



# System Architecture V1



## Design Principles

The product is built on the foundations Domain-Driven Design Principles. Taking the goals of our product, we decomposed it into a set of logical `modules`.


This document walk through the modules, their design patterns and interactions with each other. 


## Service Layer

We built the service layer with a few key technologies: 
- FastAPI for async and high performance.
- RabbitMQ for asynchronous messaging and message broker.


## Domain Layer

### Core Domain Model
The core module is called `Altas`. It is the foundation domain model, built on DDD concepts of Value Objects (e.g. unit management, chemical compounds, reaction sets), Entities (equipment, streams), Collections (parameters, properties) and Aggregate Roots (plants)

Atlas composes these units using a graph-based datastructure. The benefit of this approach is that it allows us to build a system that is agnostic to the underlying technology stack and allows natural, intuitive modelling of the domain problem.

We've invested the time and expertise to translate broad range of equipment and stream types into individual sub-classes, enabling a high degree of modularity and reusability across a range of equipment and stream implmentations.

Our compound library currently encompasses the CAS schema. 

(elaborate)
(UML on class)

### Process/AI Models & Model Optimizers (Matrix & Daedelus)

For the purpose of physics-based computation of parameters and properties,our product integrates with a host of softwares like gproms, X and Y. These all fall into the `Matrix` module, with is a manager class that implements a ports-and-adaptor pattern between `Atlas` and these process simulators.
 

(include example of internal API)

ML-based methods are used to augment physics-based methods. These are managed under a module called `Daedelus`. Under the hood, domain-relevant wrappers are built around TorchRL, PyTorch and LightningAI. Similar to `Matrix`, this module is driven by `Atlas` objects. Given the pace of AI development, we decided upon this approach to  enable us to swap out the underlying AI libraries in the future without breaking existing code.

Both Matrix and Daedelus share similar interfaces. This is an intentional decision for ease of integration with our optimizer.

### Model Optimizers (Metis)

Our optimizer module is responsible for hyperparameter optimization, experiment tracking and model management. Bundling these three responsibilities into the abstraction layer simplifies the model building process and has accelerated the model development process. Similar to Matrix, this module is wrappers built around current best-in-class solutions, with the expecations that they can be swapped out in the future without breaking existing code.

For gradient-based optimization, `Optuna` is used for its ability for paralllization and distributed computation. 

For gradient-free optimization, we use our own in-house library built on the research of Dr. Sushant Garud. 

For experiment tracking and model management, we use the open-source solution, `MLFlow`. 


### Monitoring & Analytics (Athena)

Monitoring and Analytics is a statemless module that manages compute care for a broad range of analytics: model drift, economic, (sushant to add inmore)

The module is built on as a stateless library optimized for vector operations. Unlike the other OOP-heavy modules, this module relies on FP (functional programming) principles. 

(elaborate on why)


## Data Layer

### Data Management & Pipelines

We use a repoistory pattern for data persistence.
- sqlalchemy for bla
- backend in sqllite for development and postgresql for production

For data pipelines, we implemented Prefect our scheduler and manager. We selected this sbecase (3 reasons)






![image](https://www.cosmicpython.com/book/images/apwp_1201.png)

# OLD
## Key Modules & Interactions

TLDR:


Details:

- Atlas - domain model built on graph networks
    - Defines aggregate roots, entities and value objects. 
    - The main point of interaction for other modules
- Metis - optimizer module
    - Responsible for performant search and optimization of chemical process optimization problems. Does this through use of distributed processes.
    - special libraries: 
    - Optuna for gradient-based optimization
    - In-House library for gradient-free optimization
    - MLFlow for experiment & model tracking 
- Athena - analytics module
    - noteworthy libraries:
    - LIME for feature importance visualization
    - SKLearn for diagnostics
    - Evidently for model monitoring
- Capricorn - data manager module 
    - uses a repository pattern to manage data access
    - Prefect for orchestration 
- Matrix - process modelling module
    - interfaces with 3rd party process modelling softwares for process simulation
    - uses a Ports-and-Adaptor pattern to communicate with the 3rd party software in a reusable way.
- Daedelus - AI module 
    - used to build AI and RL models
    - interfaces with Metis and Athena
    - for Neural networks, built on PyTorch and LightningAI
    - for RL, built on TorchRL
- Service Layer
    - HTTP layer that serves as entry point to app
    - uses FastAPI for async.
    - uses RabbitMQ for as message broker
- Acorn - our frontend
    - currently built on React Technologies via Dash and Dash-Bootstrap
    - Interacts via HTTP service layer using REST with the app. 

## Deep Dive: Class Diagram for Domain Model

Class

## Deep Dive: Tech Stack Diagram


```Mermaid
graph TD
    subgraph "Single DigitalOcean Droplet"
        subgraph "Application & Data Services"
            A[Docker: Core App]
            B[Docker: PostgreSQL]
            C[Docker: RabbitMQ]
            D[Docker: Frontend]
        end
        subgraph "MLOps Services"
            E[Docker: MLflow]
            F[Docker: Prefect]
        end
    end
    G[Load Balancer] --> A
    G --> D
    G --> E
    G --> F
    H[Users] --> G

```

Application & Data Server
Hosts:
- Main application logic (Atlas, Metis, Athena, Capricorn, Matrix, Daedelus)
- API endpoints (FastAPI)
- Frontend files (React)
- PostgreSQL database
- RabbitMQ message broker
- Technologies: Python, React, PostgreSQL, RabbitMQ

MLOps Server
Hosts:
- MLflow tracking server (w SQLite)
- Prefect Orion server (w SQLite)
- Technologies: MLflow, Prefect

```Mermaid
graph TD
    subgraph "Application & Data Server"
        A[Core Application] --> B[PostgreSQL]
        A --> C[RabbitMQ]
        D[Frontend] --> A
    end
    subgraph "MLOps Server"
        E[MLflow] --> F[SQLite]
        G[Prefect] --> H[SQLite]
    end
    A --> E
    A --> G

```

## Deep Dive: Optimizing Optimization 

TLDR: 



---
END 

## Network Architecture V1 - Monolithic 

- Container 1: main app
    - Contains: Atlas, Metis, Athena, Capricorn, Matrix, Daedelus, Service Layer
    - Base Image: Python 3.8
    - Exposed Ports: 8000 (FastAPI)
    - Volume: /app/data (for persistent storage)
    - Environment Variables: DB_URI, MLFLOW_TRACKING_URI
- Container 2: front end 
    - Contains: Acorn (React-based Dash application)
    - Base Image: Python 3.8 
    - Exposed Ports: 3000 (React development server)
    - Environment Variables: API_BASE_URL
- Container 3: MLFlow server, backend DB and artefact store (all in one)
    - interacts with main  app via Metis
    - kept persistent so devs can interact with it
    - Contains: MLflow server, backend DB, artifact store
    - Base Image: Python 3.9
    - Exposed Ports: 5000 (MLflow UI)
    - Volumes:
        - /mlflow/database (for SQLite)
        - /mlflow/artifacts (for artifact storage)
    - Environment Variables: MLFLOW_TRACKING_URI
- Container 4: All in One Container for Prefect 
    - Contains: Prefect server, backend DB
    - Base Image: Python 3.9
    - Exposed Ports: 4200 (Prefect UI)
    - Volumes: /prefect/database (for SQLite)
    - Environment Variables: PREFECT_SERVER_DATABASE_URL
- Database Container:
    - Contains: PostgreSQL
    - Base Image: postgres:13
    - Exposed Ports: 5432
    - Volumes: /var/lib/postgresql/data
    - Environment Variables: POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
- Message Broker Container:
    - Contains: RabbitMQ
    - Base Image: rabbitmq:3-management
    - Exposed Ports: 5672 (AMQP), 15672 (Management UI)
    - Environment Variables: RABBITMQ_DEFAULT_USER, RABBITMQ_DEFAULT_PASS


```Mermaid
graph TD
    A[Application Container] --> B[Database Container]
    A --> C[MLflow Container]
    A --> D[Prefect Container]
    A --> E[Message Broker Container]
    F[Frontend Container] --> A
    C --> B
    D --> B
    A --> G[Artifact Storage]
    C --> G

```

## Developer Productivity
 while working on a local development container, the developer can still interact with prefect and mlflow servers for centralized experiment tracking and orchaestration
 - Via Docker Network: If you're running your local development environment in Docker, you can create a Docker network that includes your local containers and has a connection to the remote services.

### Local Env <> Deployed Services
The architecture allows for multiple Prefect and MLFlow clients across different setups (e.g., different developer machines, CI/CD pipelines, or even different applications) all interacting with the same remote server. Each client can manage its own set of flows and orchestrations.

This setup offers several advantages:
- Centralized Orchestration: All flows are managed and monitored from a single Prefect server.
- Distributed Execution: Flows can be executed on different machines or environments.
- Team Collaboration: Multiple team members can work on different parts of the workflow while using the same central server.
- Environment Flexibility: You can have development, staging, and production environments all using the same Prefect server but with different agents and execution environments.

```
[Local Dev Environment 1] ----\
                               \
[Local Dev Environment 2] ------\
                                 \
[CI/CD Pipeline] ------------------> [Remote Prefect Server] <--> [Prefect UI]
                                 /
[Staging Environment] -----------/
                               /
[Production Environment] -----/

```

PYTHON ?= python
APP_PORT ?= 8501

.PHONY: install app check test australia docker-build docker-run docker-compose-up docker-compose-down

install:
	$(PYTHON) -m pip install -r requirements.txt

app:
	streamlit run app.py --server.port=$(APP_PORT)

check:
	$(PYTHON) -m py_compile app.py domain_shift_australia.py
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

australia:
	$(PYTHON) domain_shift_australia.py

docker-build:
	docker build -t car-price-app .

docker-run:
	docker run --rm --name car-price-app -p $(APP_PORT):8501 car-price-app

docker-compose-up:
	docker compose up --build

docker-compose-down:
	docker compose down

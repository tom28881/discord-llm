# Discord LLM Testing Makefile

.PHONY: help install test test-unit test-integration test-ml test-performance test-quality test-stress test-all test-fast format lint type-check security clean coverage report

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
TEST_RUNNER := python scripts/run_tests.py

help: ## Show this help message
	@echo "Discord LLM Testing Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make install       # Install all dependencies"
	@echo "  make test          # Run all tests"
	@echo "  make test-fast     # Run only fast tests"
	@echo "  make format        # Format code with Black"
	@echo "  make report        # Generate quality report"

install: ## Install all dependencies
	@echo "🔧 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	@echo "✅ Dependencies installed"

install-dev: install ## Install development dependencies
	@echo "🔧 Installing development dependencies..."
	$(PIP) install pre-commit
	pre-commit install
	@echo "✅ Development environment ready"

# Test commands
test: ## Run all tests (comprehensive)
	@echo "🧪 Running comprehensive test suite..."
	$(TEST_RUNNER) --suite all

test-fast: ## Run only fast tests
	@echo "⚡ Running fast tests..."
	$(TEST_RUNNER) --suite all --fast

test-unit: ## Run unit tests only
	@echo "🔬 Running unit tests..."
	$(TEST_RUNNER) --suite unit

test-integration: ## Run integration tests only
	@echo "🔗 Running integration tests..."
	$(TEST_RUNNER) --suite integration

test-ml: ## Run ML model tests only
	@echo "🤖 Running ML model tests..."
	$(TEST_RUNNER) --suite ml

test-performance: ## Run performance tests only
	@echo "⚡ Running performance tests..."
	$(TEST_RUNNER) --suite performance

test-quality: ## Run quality metric tests only
	@echo "📊 Running quality tests..."
	$(TEST_RUNNER) --suite quality

test-stress: ## Run stress tests only
	@echo "💪 Running stress tests..."
	$(TEST_RUNNER) --suite stress

test-all: ## Run all tests including stress tests
	@echo "🚀 Running complete test suite..."
	$(TEST_RUNNER) --suite all --include-stress

# Individual test categories with pytest
pytest-unit: ## Run unit tests with pytest directly
	$(PYTEST) tests/unit/ -v --tb=short -m "unit and not slow"

pytest-integration: ## Run integration tests with pytest directly
	$(PYTEST) tests/integration/ -v --tb=short -m "integration and not slow"

pytest-ml: ## Run ML tests with pytest directly
	$(PYTEST) tests/ml/ -v --tb=short -m "ml and not slow"

pytest-performance: ## Run performance tests with pytest directly
	$(PYTEST) tests/performance/ -v --tb=short --benchmark-only -m "performance and not stress"

# Code quality commands
format: ## Format code with Black
	@echo "🎨 Formatting code..."
	black .
	@echo "✅ Code formatted"

format-check: ## Check code formatting
	@echo "🔍 Checking code formatting..."
	black --check --diff .

lint: ## Run linting with Flake8
	@echo "🔍 Running linter..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "✅ Linting complete"

lint-full: ## Run comprehensive linting
	@echo "🔍 Running comprehensive linting..."
	flake8 . --count --statistics

type-check: ## Run type checking with MyPy
	@echo "🔍 Running type checking..."
	mypy lib/ --ignore-missing-imports
	@echo "✅ Type checking complete"

security: ## Run security scans
	@echo "🔒 Running security scans..."
	bandit -r lib/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	@echo "✅ Security scans complete"

quality-all: format-check lint type-check security ## Run all quality checks

# Coverage commands
coverage: ## Run tests with coverage
	@echo "📊 Running tests with coverage..."
	$(PYTEST) tests/ --cov=lib --cov-report=html:htmlcov --cov-report=term-missing --cov-report=xml

coverage-unit: ## Run unit tests with coverage
	$(PYTEST) tests/unit/ --cov=lib --cov-report=html:htmlcov-unit --cov-report=term-missing

coverage-integration: ## Run integration tests with coverage
	$(PYTEST) tests/integration/ --cov=lib --cov-report=html:htmlcov-integration --cov-report=term-missing

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "⚡ Running performance benchmarks..."
	$(PYTEST) tests/performance/ --benchmark-only --benchmark-json=benchmark-results.json

# Quality metrics and reporting
report: ## Generate comprehensive quality report
	@echo "📋 Generating quality report..."
	$(PYTHON) -c "from tests.utils.quality_metrics import get_quality_collector; collector = get_quality_collector(); report = collector.generate_quality_report(); print(f'Quality Score: {report.overall_score:.2f}')"

metrics: ## Display current quality metrics
	@echo "📊 Current Quality Metrics:"
	@$(PYTHON) -c "from tests.utils.quality_metrics import get_quality_collector; collector = get_quality_collector(); stats = collector.get_test_results_summary(); print(f'Tests: {stats.get(\"total\", 0)} total')"

export-metrics: ## Export quality metrics to JSON
	@echo "📤 Exporting quality metrics..."
	$(PYTHON) -c "from tests.utils.quality_metrics import get_quality_collector; collector = get_quality_collector(); collector.export_metrics('quality-metrics-export.json')"
	@echo "✅ Metrics exported to quality-metrics-export.json"

# Database and cleanup
clean: ## Clean up test artifacts and cache
	@echo "🧹 Cleaning up..."
	rm -rf .pytest_cache/
	rm -rf htmlcov*/
	rm -rf .coverage
	rm -rf *.xml
	rm -rf *.json
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

clean-db: ## Clean up test databases
	@echo "🗄️ Cleaning test databases..."
	rm -rf data/test_*.db
	rm -rf data/*test*.sqlite
	@echo "✅ Test databases cleaned"

# CI/CD simulation
ci-simulation: ## Simulate CI/CD pipeline locally
	@echo "🔄 Simulating CI/CD pipeline..."
	@echo "1. Quality checks..."
	@$(MAKE) quality-all
	@echo "2. Unit tests..."
	@$(MAKE) test-unit
	@echo "3. Integration tests..."
	@$(MAKE) test-integration  
	@echo "4. ML tests..."
	@$(MAKE) test-ml
	@echo "5. Performance tests..."
	@$(MAKE) test-performance
	@echo "6. Generating report..."
	@$(MAKE) report
	@echo "✅ CI/CD simulation complete"

# Development workflow
dev-setup: install-dev ## Complete development setup
	@echo "🚀 Development environment setup complete!"
	@echo "Next steps:"
	@echo "  - Run 'make test-fast' to verify setup"
	@echo "  - Use 'make test' for comprehensive testing"
	@echo "  - Use 'make format' before committing code"

dev-test: format-check test-unit test-integration ## Quick development test cycle
	@echo "✅ Development test cycle complete"

pre-commit: format lint type-check test-unit ## Pre-commit checks
	@echo "✅ Pre-commit checks passed"

# Documentation
docs: ## Generate test documentation
	@echo "📚 Generating test documentation..."
	$(PYTEST) tests/ --collect-only --quiet | grep -E "^<.*>" | wc -l | xargs -I {} echo "Total tests: {}"
	@echo "Test categories:"
	@$(PYTEST) tests/ --collect-only --quiet | grep -o "test_[a-zA-Z_]*" | sort | uniq -c | sort -nr

# Monitoring
monitor-start: ## Start quality monitoring
	@echo "👁️ Starting quality monitoring..."
	$(PYTHON) -c "from tests.utils.quality_metrics import get_quality_collector, QualityMonitor; collector = get_quality_collector(); monitor = QualityMonitor(collector); monitor.start_monitoring(); print('Quality monitoring started')"

# Release preparation
release-check: clean quality-all test coverage ## Full release readiness check
	@echo "🎯 Release readiness check complete"
	@$(MAKE) report

# Performance profiling
profile: ## Run performance profiling
	@echo "📈 Running performance profiling..."
	$(PYTHON) -m cProfile -o profile-stats.prof -c "import subprocess; subprocess.run(['pytest', 'tests/performance/', '--benchmark-only'])"
	@echo "✅ Profile saved to profile-stats.prof"

# Help with common workflows
workflow-help: ## Show common workflow examples
	@echo "🔄 Common Testing Workflows:"
	@echo ""
	@echo "Development Workflow:"
	@echo "  make dev-setup     # First time setup"
	@echo "  make dev-test      # Quick test during development"
	@echo "  make pre-commit    # Before committing changes"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test          # Comprehensive testing"
	@echo "  make coverage      # Test coverage analysis"
	@echo "  make report        # Quality metrics report"
	@echo ""
	@echo "Performance Analysis:"
	@echo "  make benchmark     # Performance benchmarks"
	@echo "  make test-stress   # Stress testing"
	@echo "  make profile       # Performance profiling"
	@echo ""
	@echo "Release Preparation:"
	@echo "  make release-check # Full release readiness"
	@echo "  make ci-simulation # Simulate CI/CD pipeline"
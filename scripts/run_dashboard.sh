

set -e

echo "🚧 Building and starting all services (API, ML, Dashboard)..."
docker-compose up --build -d

echo
echo "🔗 Services are up! Access them at:"
echo "  • API service:           http://localhost:8000"
echo "  • ML inference service:  http://localhost:5000"
echo "  • Streamlit dashboard:   http://localhost:8501"
echo
echo "To view logs, run: docker-compose logs -f"

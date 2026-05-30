# Social Media Emotion Recognition

Production-oriented monorepo for English social media emotion recognition. The app accepts informal text, slang, emojis, abbreviations, and noisy comments, then returns the predicted emotion, display labels, emoji, confidence, language, model metadata, and full probability distribution.

## Architecture

Stage 1 and Stage 2 are implemented as a 3-tier client-server application with a separated model inference service:

```text
Next.js frontend -> Node.js Express API -> FastAPI model registry -> PostgreSQL
```

Batch CSV processing is prepared with Redis and BullMQ:

```text
Next.js frontend -> Node.js Express API -> Redis/BullMQ -> Worker -> FastAPI model API -> PostgreSQL
```

Clean Architecture is intentionally not used. The code follows a pragmatic route-controller-service structure suitable for this project size.

## Tech Stack

- Frontend: Next.js App Router, TypeScript, Tailwind CSS, shadcn-style UI components, React Hook Form, Zod, TanStack Query, Recharts.
- Backend: Node.js, Express.js, TypeScript, Zod, Prisma, PostgreSQL, Winston, Morgan, Helmet, CORS, rate limiting, Multer, BullMQ.
- Model API: FastAPI, Uvicorn, PyTorch, Transformers, Pydantic, python-dotenv, Vietnamese PhoBERT and English RoBERTa models.
- Infrastructure: PostgreSQL and Redis via Docker Compose for local development.

## Local Development

Start PostgreSQL and Redis:

```bash
docker compose up -d
```

Start the model API:

```bash
cd apps/model-api
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Start the backend:

```bash
cd apps/api
npm install
npm run prisma:generate
npm run prisma:migrate
npm run dev
```

Start the frontend:

```bash
cd apps/web
npm install
npm run dev
```

From the monorepo root, install all Node workspaces with:

```bash
npm install
```

Open [http://localhost:3000](http://localhost:3000).

## Health Checks

- Model API: [http://localhost:8000/health](http://localhost:8000/health)
- Node API: [http://localhost:4000/health](http://localhost:4000/health)

## Testing

```bash
npm run lint
npm run typecheck
npm run build
npm run test
```

For the model service:

```bash
cd apps/model-api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then send Vietnamese examples to `POST /predict`.

## Deployment

- Frontend: Vercel, project directory `emotion-recognition-app/apps/web`, env `NEXT_PUBLIC_API_URL`.
- Backend: Render, Railway, Fly.io, VPS, or equivalent, env `DATABASE_URL`, `MODEL_API_URL`, `CORS_ORIGIN`, `REDIS_URL`.
- Model API: Render, Railway, Hugging Face Spaces, VPS, or GPU server. Cold starts should account for model download/load time.
- Database: Supabase, Neon, Railway PostgreSQL, or equivalent.
- Redis: Upstash, Railway Redis, Redis Cloud, or local Docker.

See `docs/` for detailed specifications and operations notes.

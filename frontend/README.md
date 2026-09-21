# GeoVisionGPT — frontend

Next.js (App Router) + Tailwind. Talks to the FastAPI backend in `../backend`.

```bash
npm install
npm run dev        # http://localhost:3000
```

The backend must be running (default `http://localhost:8000`); point at another one with
`NEXT_PUBLIC_API_URL=http://host:port/api/v1`. The backend's CORS list already allows
`http://localhost:3000`.

## What's here

| Route | Purpose |
|---|---|
| `/login` | Sign in / create a researcher account |
| `/images` | Image library — drag-and-drop upload, thumbnails, metadata, delete |
| `/images/[id]` | Workspace — ask a question, see both models' answers side by side, question history |

## Where things live

- `src/lib/api.ts` — every backend call, bearer-token handling, and single-flight refresh
  (refresh tokens are single-use server-side, so concurrent 401s must share one refresh).
- `src/lib/types.ts` — mirrors `backend/app/application/dto.py`.
- `src/lib/models.ts` — labels/colours for the two fusion variants and the suggested questions.
- `src/components/AnswerComparison.tsx` — the side-by-side answer cards.

// Adresa backend-ului WaferSight (FastAPI), fără "/" la final.
// Gol = UI-ul e servit chiar de backend (uvicorn api.main:app), ca local.
// Pe un host doar pentru frontend (Cloudflare Pages, Netlify...) pune aici URL-ul backend-ului, ex. "https://wafersight-api.onrender.com".
// Se poate suprascrie din browser cu ?api=https://... (ținut minte; ?api= gol revine la valoarea de aici).
window.WAFERSIGHT_API = "";

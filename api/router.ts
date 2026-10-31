import { authRouter } from "./auth-router";
import { experimentRouter } from "./experiment-router";
import { createRouter, publicQuery } from "./middleware";

export const appRouter = createRouter({
  ping: publicQuery.query(() => ({ ok: true, ts: Date.now() })),
  auth: authRouter,
  experiment: experimentRouter,
});

export type AppRouter = typeof appRouter;

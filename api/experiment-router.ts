import { z } from "zod";
import { authedQuery, createRouter } from "./middleware";
import { createExperiment, deleteExperiment, listExperiments } from "./queries/experiments";

export const experimentRouter = createRouter({
  list: authedQuery.query(({ ctx }) => listExperiments(ctx.user.id)),
  create: authedQuery
    .input(z.object({
      title: z.string().trim().min(1).max(120),
      prompt: z.string().trim().min(1).max(80),
      output: z.string().max(500),
      temperature: z.number().min(0.1).max(2),
      topK: z.number().int().min(1).max(29),
    }))
    .mutation(({ ctx, input }) => createExperiment({ ...input, userId: ctx.user.id })),
  delete: authedQuery
    .input(z.object({ id: z.number().int().positive() }))
    .mutation(({ ctx, input }) => deleteExperiment(input.id, ctx.user.id)),
});

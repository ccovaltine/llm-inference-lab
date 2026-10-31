import { and, desc, eq } from "drizzle-orm";
import { experiments } from "@db/schema";
import { getDb } from "./connection";

export async function listExperiments(userId: number) {
  return getDb()
    .select()
    .from(experiments)
    .where(eq(experiments.userId, userId))
    .orderBy(desc(experiments.createdAt))
    .limit(30);
}

export async function createExperiment(data: {
  userId: number;
  title: string;
  prompt: string;
  output: string;
  temperature: number;
  topK: number;
}) {
  const [{ id }] = await getDb().insert(experiments).values(data).$returningId();
  return getDb().query.experiments.findFirst({ where: eq(experiments.id, id) });
}

export async function deleteExperiment(id: number, userId: number) {
  await getDb().delete(experiments).where(and(eq(experiments.id, id), eq(experiments.userId, userId)));
  return { ok: true };
}

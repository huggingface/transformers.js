import { json } from '@sveltejs/kit';
import PipelineSingleton from '$lib/server/pipeline.js';

export async function GET({ url }) {
  const text = url.searchParams.get('text');
  if (!text) {
    return json(
      {
        error: 'Missing text parameter'
      },
      { status: 400 }
    );
  }
  // Get the classification pipeline. When called for the first time,
  // this will load the pipeline and cache it for future use.
  const classifier = await PipelineSingleton.getInstance();

  // Actually perform the classification
  const result = await classifier(text);

  return json(result);
}

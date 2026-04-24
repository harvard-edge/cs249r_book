const BASEURL = `https://letmeeatkaik.azurewebsites.net/`
const TESTURL = `http://localhost:3000/`

const jsonUrl = `${BASEURL}api/lang-groq`;

// import {promptRadial} from '../configs/agents.config.js';



export async function* jsonAgent(text) {
  // TRACE: Track which AI endpoint is being called
  console.trace(`[AI_TRACE] json_streamer_agent.js - Calling JSON agent endpoint: ${jsonUrl}`, {
    textPreview: text?.substring(0, 100) + '...' || 'No text'
  });

  const response = await fetch(jsonUrl, {
      method: 'POST',
      body: JSON.stringify({ text: text}),
      headers: { 'Content-Type': 'application/json' }
  });

  console.log(response);

  if (!response.ok) {
      console.error('Network response was not ok');
      return;
  }

  const reader = response.body.getReader();

  const stream = new ReadableStream({
      start(controller) {
          function push() {
              reader.read().then(({ done, value }) => {
                  if (done) {
                      controller.close();
                      return;
                  }

                  controller.enqueue(value);
                  push();
              });
          } 

          push();
      }
  });

  const reader2 = stream.getReader();

  while (true) {
      const { done, value } = await reader2.read();

      if (done) {
          break;
      }

      yield value;
  }
}
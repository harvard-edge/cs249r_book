import { VectorDB, getDBDefaults, setDBDefaults } from "./db/vectorDB/vectordb";
import { CustomTextSplitter } from "./splitters/text_splitter";
// import { embed } from "./embeddings/xenova_embeddings_gte";
import {fetchEmbeddings} from './embeddings/api_call_embeddings'

const intervalSize = 200; // Example interval size
const overlapSize = 20; // Example overlap size


const textSplitter = new CustomTextSplitter(intervalSize, overlapSize, '\n\n');



// const db = new VectorDB({
//   vectorPath: "embedding"
// });

let db;
const inMemoryRecord = [];

export async function initiate(dbName)
{
  // const res = await embed("n")
  // return res
  // DB_DEFAULTS.dbName = dbName
  setDBDefaults({ dbName: dbName });

  db = new VectorDB({
    vectorPath: "embedding"
  });

  
  return getDBDefaults()
}


export async function quickStart(metaData, token, isLocal = true) {

  // const text = metaData.text;
  const { text, ...otherMetaData } = metaData;
  const chunks = textSplitter.split(text);

  // Map each chunk to a promise that resolves to its embedding

  // Await all embedding promises
  let embeddings;
  // if(isLocal){
  //   const embeddingPromises = chunks.map(chunk => embed(chunk));

  //  embeddings = await Promise.all(embeddingPromises);
  // }
  // else {
    embeddings = await fetchEmbeddings(chunks, token)
  // }


  for (let i = 0; i < chunks.length; i++) {
    const { text, ...otherMetaData } = metaData;
    otherMetaData.text = chunks[i]; // Assign chunks[i] to otherMetaData with the key 'text'
    const key = await db.insert({
        embedding: embeddings[i], // Use the awaited embedding
        ...otherMetaData // Spread the otherMetaData object, which now includes the 'text' key
    });
    inMemoryRecord.push({
      key: key,
      // embedding: embeddings[i], // Use the awaited embedding
      ...otherMetaData 
    });
  }
  return inMemoryRecord;
}


export async function search(query,  k={ limit: 5 },token, isLocal=true) {
  let queryEmbeddings;
  // if (isLocal)
  //  queryEmbeddings = await embed(query)
  // else {
    queryEmbeddings = await fetchEmbeddings(query, token)

  // }
  const results = await db.query(queryEmbeddings, k);


  return results
}



export async function quickStart_single(metaData, token, isLocal = true) {

  // const text = metaData.text;
  const { text, ...otherMetaData } = metaData;

  // Map each chunk to a promise that resolves to its embedding

  // Await all embedding promises
  // if(isLocal){
    // const embeddingPromises = chunks.map(chunk => embed(chunk));

  // const embeddings = await embed(text) // await Promise.all(embeddingPromises);
  const embeddings = fetchEmbeddings([text], token)


  // for (let i = 0; i < chunks.length; i++) {
  //   const { text, ...otherMetaData } = metaData;
  //   otherMetaData.text = chunks[i]; // Assign chunks[i] to otherMetaData with the key 'text'
    const key = await db.insert({
        embedding: embeddings, // Use the awaited embedding
        text,
        ...otherMetaData // Spread the otherMetaData object, which now includes the 'text' key
    });
    inMemoryRecord.push({
      key: key,
      // embedding: embeddings[i], // Use the awaited embedding
      ...otherMetaData 
    });
  return inMemoryRecord;
}


// export async function search(query,  k={ limit: 5 },token, isLocal=true) {
//   let queryEmbeddings;
//   if (isLocal)
//    queryEmbeddings = await embed(query)
//   else {
//     queryEmbeddings = await fetchEmbeddings([query], token)
//   }
//   const results = await db.query(queryEmbeddings, k);
//   return results
// }



export async function delete_row(key){
    return await db.delete(key)
}

export async function update(key, metaData, token) {
  const text = metaData.text

  if(!metaData.embedding){
  const text_embeddings = await fetchEmbeddings([text], token)
    metaData.embedding = text_embeddings
}

  // const embeddings = fetchEmbeddings([text], token)

  return await db.update(key, {embedding: text_embeddings,  metaData: metaData}); //"text": text})
}
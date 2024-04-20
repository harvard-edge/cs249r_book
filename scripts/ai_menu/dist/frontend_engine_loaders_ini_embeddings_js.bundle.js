/*
 * ATTENTION: The "eval" devtool has been used (maybe by default in mode: "development").
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(self["webpackChunkinjectchat"] = self["webpackChunkinjectchat"] || []).push([["frontend_engine_loaders_ini_embeddings_js"],{

/***/ "./frontend_engine/db/database.js":
/*!****************************************!*\
  !*** ./frontend_engine/db/database.js ***!
  \****************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   IndexedDBHelper: () => (/* binding */ IndexedDBHelper),\n/* harmony export */   databaseIndicesClassSingleton: () => (/* binding */ databaseIndicesClassSingleton)\n/* harmony export */ });\n/* harmony import */ var idb__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! idb */ \"./node_modules/idb/build/index.js\");\n\r\n\r\n\r\n// async function createDatabase(embeddings) {\r\n//   const db = await openDB('tinyML', 1, {\r\n//     upgrade(db) {\r\n//       const store = db.createObjectStore('embeddings', {\r\n//         keyPath: 'id',\r\n//         autoIncrement: true,\r\n//       });\r\n//       store.createIndex('embedding', 'embedding', { unique: false });\r\n//     },\r\n//   });\r\n\r\n//   // Store embeddings\r\n//   const tx = db.transaction('embeddings', 'readwrite');\r\n//   for (const item of embeddings) {\r\n//     tx.store.add({ text: item.text, embedding: item.embedding });\r\n//   }\r\n//   await tx.done;\r\n// }\r\n\r\n\r\nclass IndexedDBHelper {\r\n  static instance = null;\r\n  constructor(databaseName, tableName) {\r\n    if (!IndexedDBHelper.instance) {\r\n        this.dbName = databaseName;\r\n        this.tableName = tableName;\r\n        this.db = null;\r\n        this.keys = [];\r\n        IndexedDBHelper.instance = this;\r\n    }\r\n    return IndexedDBHelper.instance;\r\n}\r\n\r\n  // Initialize the database connection\r\n  async init() {\r\n    if (!this.db) {\r\n        this.db = await (0,idb__WEBPACK_IMPORTED_MODULE_0__.openDB)(this.dbName, 1, {\r\n            upgrade: (db) => {\r\n                if (!db.objectStoreNames.contains(this.tableName)) {\r\n                    db.createObjectStore(this.tableName, { keyPath: 'id' });\r\n                }\r\n            },\r\n        });\r\n        await this._loadKeys();\r\n    }\r\n}\r\n\r\n  // Load and store the keys from the IndexedDB table\r\n  async _loadKeys() {\r\n    const tx = this.db.transaction(this.tableName, 'readonly');\r\n    const store = tx.objectStore(this.tableName);\r\n    const keyList = await store.getAllKeys();\r\n    this.keys = keyList.map(key => parseFloat(key)).filter(key => !isNaN(key));\r\n  }\r\n\r\n  // Find the k closest keys to the given float\r\n  findClosestKeys(targetFloat, k) {\r\n    const sortedDiffs = this.keys\r\n      .map(key => ({ key, diff: Math.abs(targetFloat - key) }))\r\n      .sort((a, b) => a.diff - b.diff);\r\n    return sortedDiffs.slice(0, k).map(item => item.key);\r\n  }\r\n\r\n  // Retrieve rows based on an array of keys\r\n  async getRows(keys) {\r\n    const tx = this.db.transaction(this.tableName, 'readonly');\r\n    const store = tx.objectStore(this.tableName);\r\n    const results = [];\r\n\r\n    for (const key of keys) {\r\n      const request = store.get(key.toString());\r\n      const result = await request;\r\n      if (result) {\r\n        results.push(result);\r\n      } else {\r\n        console.error(`Row with key ${key} doesn't exist.`);\r\n      }\r\n    }\r\n\r\n    return results;\r\n  }\r\n\r\n    // New method to get all keys from the table\r\n    async getAllKeys() {\r\n      if (!this.db) {\r\n        await this.init();\r\n      }\r\n      const tx = this.db.transaction(this.tableName, 'readonly');\r\n      const store = tx.objectStore(this.tableName);\r\n      return await store.getAllKeys();\r\n    }\r\n\r\n      // Static method to get the instance\r\n // Static method to get the instance\r\nstatic getInstance(databaseName, tableName) {\r\n  if (IndexedDBHelper.instance === null) {\r\n    IndexedDBHelper.instance = new IndexedDBHelper(databaseName, tableName);\r\n    IndexedDBHelper.instance.init(); // Initialize right after creation\r\n  }\r\n  return IndexedDBHelper.instance;\r\n}\r\n\r\n}\r\n\r\n class SingletonWrapper {\r\n  constructor() {\r\n      this._instancePromise = null;\r\n  }\r\n\r\n  async initInstance(databaseName, tableName) {\r\n      if (!this._instancePromise) {\r\n          this._instancePromise = (async () => {\r\n              const dbHelper = new IndexedDBHelper(databaseName, tableName);\r\n              await dbHelper.init();\r\n              return dbHelper;\r\n          })();\r\n      }\r\n      return this._instancePromise;\r\n  }\r\n\r\n  async getInstance() {\r\n      if (!this._instancePromise) {\r\n          throw new Error(\"SingletonWrapper instance is not initialized. Call initInstance first.\");\r\n      }\r\n      return this._instancePromise;\r\n  }\r\n}\r\n\r\nconst databaseIndicesClassSingleton = new SingletonWrapper();\r\n\r\n\r\n// // use this to get singleton\r\n// export async function useDatabaseHelperSafely(arrayOfRows=null) {\r\n//   try {\r\n//       const dbHelper = await databaseIndicesClassSingleton.getInstance();\r\n//       // Now use dbHelper to interact with the database\r\n//       // For example, to get rows:\r\n//       if(arrayOfRows) {\r\n//       const rows = await dbHelper.getRows(arrayOfRows);\r\n//       console.log(rows);\r\n//       }\r\n//       else{\r\n//         const allKeys = await dbHelper.getAllKeys();\r\n//         console.log(\"All keys:\", allKeys);\r\n//       }\r\n//   } catch (error) {\r\n//       console.error(\"Database helper is not initialized:\", error);\r\n//       // Handle not-initialized case, maybe retry or initialize\r\n//   }\r\n// }\r\n\r\n\r\n\r\n// // Usage example\r\n// // Ensure you have `idb` library installed or included in your project\r\n// const dbHelper = new IndexedDBHelper('myDatabase', 'myTable');\r\n// dbHelper.init().then(() => {\r\n//   console.log('Database initialized.');\r\n\r\n//   // Example: Find the 3 closest keys to 5.2\r\n//   const closestKeys = dbHelper.findClosestKeys(5.2, 3);\r\n//   console.log('Closest Keys:', closestKeys);\r\n\r\n//   // Example: Get rows based on keys\r\n//   dbHelper.getRows(closestKeys).then(rows => {\r\n//     console.log('Rows:', rows);\r\n//   });\r\n// });\r\n\n\n//# sourceURL=webpack://injectchat/./frontend_engine/db/database.js?");

/***/ }),

/***/ "./frontend_engine/embeddings/open_ai_embeddings.js":
/*!**********************************************************!*\
  !*** ./frontend_engine/embeddings/open_ai_embeddings.js ***!
  \**********************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   sendTextForEmbeddings: () => (/* binding */ sendTextForEmbeddings)\n/* harmony export */ });\n\r\n\r\nfunction sendTextForEmbeddings(token, textChunks) {\r\n    // Get text from textarea and split into chunks\r\n    // const text = document.getElementById('textChunksInput').value;\r\n    // const textChunks = text.split('\\n').filter(chunk => chunk.trim() !== ''); // Split by newline and filter out empty lines\r\n    // const openAiApiKey = 'Your-OpenAI-API-Key-Here'; // Ideally, this should not be exposed in the frontend for security reasons\r\n\r\n    // Define the URL of your endpoint\r\n    const url = 'http://localhost:3000/embeddings';\r\n\r\n    // Setup the request options\r\n    const options = {\r\n        method: 'POST',\r\n        headers: {\r\n            Authorization: `Bearer ${token}`,\r\n            \"Content-Type\": \"application/json\",\r\n          },\r\n        body: JSON.stringify({\r\n            textChunks: textChunks,\r\n        })\r\n    };\r\n\r\n    // Send the request\r\n    fetch(url, options)\r\n        .then(response => response.json())\r\n        .then(data => {\r\n            console.log('Embeddings:', data.embeddings);\r\n            // Handle the embeddings data here (e.g., display in UI)\r\n        })\r\n        .catch(error => {\r\n            console.error('Error fetching embeddings:', error);\r\n            // Handle request errors here (e.g., display error message)\r\n        });\r\n}\r\n\n\n//# sourceURL=webpack://injectchat/./frontend_engine/embeddings/open_ai_embeddings.js?");

/***/ }),

/***/ "./frontend_engine/embeddings/xenova_embeddings.js":
/*!*********************************************************!*\
  !*** ./frontend_engine/embeddings/xenova_embeddings.js ***!
  \*********************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   embed: () => (/* binding */ embed),\n/* harmony export */   embedWithTimeout: () => (/* binding */ embedWithTimeout)\n/* harmony export */ });\n/* harmony import */ var _xenova_transformers__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @xenova/transformers */ \"./node_modules/@xenova/transformers/src/transformers.js\");\n\r\n\r\n\r\n// TESTING\r\n// (async () => {\r\n//     const transformers = await import('@xenova/transformers');\r\n//     const { pipeline, env } = transformers;\r\n\r\n\r\n// Skip initial check for local models, since we are not loading any local models.\r\n_xenova_transformers__WEBPACK_IMPORTED_MODULE_0__.env.allowLocalModels = false;\r\n\r\n// Due to a bug in onnxruntime-web, we must disable multithreading for now.\r\n// See https://github.com/microsoft/onnxruntime/issues/14445 for more information.\r\n_xenova_transformers__WEBPACK_IMPORTED_MODULE_0__.env.backends.onnx.wasm.numThreads = 1;\r\n\r\n\r\nclass PipelineSingleton {\r\n\r\n    static task = \"feature-extraction\"// 'text-classification';\r\n    static model = \"Supabase/gte-small\"// 'Xenova/distilbert-base-uncased-finetuned-sst-2-english';\r\n    static instance = null;\r\n\r\n    static async getInstance(progress_callback = null) {\r\n        if (this.instance === null) {\r\n            this.instance = await (0,_xenova_transformers__WEBPACK_IMPORTED_MODULE_0__.pipeline)(\"feature-extraction\", \"Supabase/gte-small\", { progress_callback }); // pipeline(this.task, this.model, { progress_callback });\r\n        }\r\n\r\n        return this.instance;\r\n    }\r\n}\r\nconst embed = async (text) => {\r\n    let model;\r\n    try {\r\n        model = await PipelineSingleton.getInstance((data) => {\r\n            // Handle pipeline creation progress\r\n            console.log('Progress:', data); // this will show progress\r\n        });\r\n    } catch (error) {\r\n        console.error(\"Error initializing model:\", error);\r\n        throw error; // Re-throw to signal error condition to caller\r\n    }\r\n\r\n    try {\r\n        const embeddings = await model(text, {\r\n            pooling: \"mean\",\r\n            normalize: true,\r\n        });\r\n        return await embeddings;\r\n    } catch (error) {\r\n        console.error(\"Error generating embeddings:\", error);\r\n        // Optionally, handle the error by returning a default value or similar\r\n        throw error; // Re-throw to signal error condition to caller\r\n    }\r\n};\r\n\r\n\r\nfunction withTimeout(promise, timeoutMs, timeoutError = new Error('Operation timed out')) {\r\n    let timeoutHandle;\r\n    const timeoutPromise = new Promise((resolve, reject) => {\r\n        timeoutHandle = setTimeout(() => reject(timeoutError), timeoutMs);\r\n    });\r\n    return Promise.race([promise, timeoutPromise]).then((result) => {\r\n        clearTimeout(timeoutHandle);\r\n        return result;\r\n    }, (error) => {\r\n        clearTimeout(timeoutHandle);\r\n        throw error;\r\n    });\r\n}\r\n\r\n const embedWithTimeout = async (text, timeoutMs = 10000) => { // Default timeout of 10 seconds\r\n    try {\r\n        const model = await PipelineSingleton.getInstance();\r\n        return await withTimeout(\r\n            model(text, { pooling: \"mean\", normalize: true }),\r\n            timeoutMs\r\n        );\r\n    } catch (error) {\r\n        console.error(\"Error in embedWithTimeout:\", error);\r\n        // Optionally, handle the error by returning a default value or similar\r\n        throw error; // Or return a default value\r\n    }\r\n};\r\n\r\n\r\n// TESTING\r\n// async function runEmbedding() {\r\n//     const text = \"a single task like cooking dinner . From the moment you decide what to make , different regions of your brain , collectively referred to as the cognitive control network , collaborate to make it happen , said Anthony Wagner , a professor of psychology at Stanford and the deputy director of the university ' s Wu Tsai Neurosciences Institute . This network includes areas of your brain that are involved in executive function , or the ability to plan and carry out goal - oriented behavior . Together they\";\r\n//     try {\r\n//         const embeddings = await embed(text);\r\n//         console.log(\"Embeddings:\", embeddings);\r\n//     } catch (error) {\r\n//         console.error(\"Error generating embeddings:\", error);\r\n//     }\r\n// }\r\n// runEmbedding();\r\n\r\n// })()\n\n//# sourceURL=webpack://injectchat/./frontend_engine/embeddings/xenova_embeddings.js?");

/***/ }),

/***/ "./frontend_engine/loaders/ini_embeddings.js":
/*!***************************************************!*\
  !*** ./frontend_engine/loaders/ini_embeddings.js ***!
  \***************************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   create_embeddings: () => (/* binding */ create_embeddings),\n/* harmony export */   create_embeddings_open_ai: () => (/* binding */ create_embeddings_open_ai),\n/* harmony export */   initializeDatabase: () => (/* binding */ initializeDatabase)\n/* harmony export */ });\n/* harmony import */ var _embeddings_xenova_embeddings_js__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! ../embeddings/xenova_embeddings.js */ \"./frontend_engine/embeddings/xenova_embeddings.js\");\n/* harmony import */ var _db_database_js__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! ../db/database.js */ \"./frontend_engine/db/database.js\");\n/* harmony import */ var _embeddings_open_ai_embeddings_js__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! ../embeddings/open_ai_embeddings.js */ \"./frontend_engine/embeddings/open_ai_embeddings.js\");\n/* harmony import */ var _src_js_utils_utils_js__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! ../../src/js/utils/utils.js */ \"./src/js/utils/utils.js\");\n/* harmony import */ var _splitters_split_js__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ../splitters/split.js */ \"./frontend_engine/splitters/split.js\");\n/* harmony import */ var idb__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! idb */ \"./node_modules/idb/build/index.js\");\n\r\n// import { embed } from '../embeddings/xenova_embeddings.js';\r\n// import { CustomTextSplitter } from '../splitters/split.js';\r\n// import { openDB } from 'idb'; // Ensure you have `idb` library installed or included\r\n// import {IndexedDBHelper, databaseIndicesClassSingleton} from '../db/database.js'\r\n\r\n// export let database_indices_class;\r\n// export async function create_embeddings(text, websiteUrl) {\r\n//   console.log(\"i am getting embeddings\", text)\r\n//   text = \"This text is from the website: \" + websiteUrl + '\\n\\n' + text\r\n//   // Assuming websiteUrl is sanitized and safe to use as a store name\r\n//   const storeName = `embeddings_${websiteUrl.replace(/[^a-zA-Z0-9]/g, '_')}`; // Sanitize URL to create a valid store name\r\n//   const splitter = new CustomTextSplitter(512, 20); // Adjust interval and overlap as needed\r\n//   const chunks = splitter.split(text);\r\n\r\n//   console.log(\"I am all the website text in chunky monkey\", chunks)\r\n//   const embeddings = await Promise.all(chunks.map((chunk) =>\r\n//   embed(chunk)\r\n//     .then(tensorEmbedding => {\r\n//       // process embedding\r\n//       const embeddingArray = Array.from(tensorEmbedding.data);\r\n//       const centroid = embeddingArray.reduce((acc, val) => acc + val, 0) / embeddingArray.length;\r\n//       return {\r\n//         text: chunk,\r\n//         centroid: centroid.toFixed(15),\r\n//         embedding: embeddingArray\r\n//       };\r\n//     })\r\n//     .catch(error => {\r\n//       console.error(\"Error processing chunk:\", chunk, error);\r\n//       return null; // Handle error appropriately\r\n//     })\r\n// ));\r\n\r\n//   console.log(\"i am dbHelper\" ,embeddings)\r\n  \r\n\r\n//   await setupIndexedDB(storeName, embeddings);\r\n//   const dbHelper = await databaseIndicesClassSingleton.initInstance('tinyMl', storeName);\r\n// }\r\n\r\n// async function setupIndexedDB(storeName, embeddings) {\r\n//   const db = await openDB('tinyMl', 1, {\r\n//     upgrade(db, oldVersion, newVersion, transaction) {\r\n//       if (!db.objectStoreNames.contains(storeName)) {\r\n//         const store = db.createObjectStore(storeName, {\r\n//           keyPath: 'centroid'\r\n//         });\r\n//       }\r\n//     },\r\n//   });\r\n\r\n//   const tx = db.transaction(storeName, 'readwrite');\r\n//   embeddings.forEach(({ text, centroid }) => {\r\n//     tx.store.put({ centroid, text });\r\n//   });\r\n//   await tx.done;\r\n\r\n//   // console.log(\"we initialized the setupIndexDB\")\r\n//   // await databaseIndicesClassSingleton.initInstance('tinyMl', storeName);\r\n//   // database_indices_class.register.init()\r\n// }\r\n\r\n\r\n\r\n// import { embedSentencesWithSingleton  } from \"../embeddings/tf_embeddings.js\";\r\n\r\n// Import the SingletonWrapper and database helper\r\n\r\n\r\n\r\n\r\n\r\n\r\n // Ensure you have `idb` library installed or included\r\n\r\nasync function create_embeddings(text, websiteUrl) {\r\n  // Assuming websiteUrl is sanitized and safe to use as a store name\r\n  const storeName = `embeddings_${websiteUrl.replace(/[^a-zA-Z0-9]/g, '_')}`; // Sanitize URL to create a valid store name\r\n  const splitter = new _splitters_split_js__WEBPACK_IMPORTED_MODULE_4__.CustomTextSplitter(512, 20); // Adjust interval and overlap as needed\r\n  const chunks = splitter.split(text);\r\n  \r\n  const embeddings = await Promise.all(chunks.map(async (chunk) => {\r\n    const tensorEmbedding = await (0,_embeddings_xenova_embeddings_js__WEBPACK_IMPORTED_MODULE_0__.embed)(chunk);\r\n\r\n    \r\n    // Assuming tensorEmbedding.data contains the Float32Array of embedding values\r\n    const embeddingArray = Array.from(tensorEmbedding.data); // Convert Float32Array to regular array\r\n    // Now you can use .reduce on the array\r\n    let centroid = embeddingArray.reduce((acc, val) => acc + val, 0) / embeddingArray.length;\r\n    // centroid = !isNaN(centroid) && isFinite(centroid) ? centroid.toFixed(10) : \"fallbackUniqueKey\";\r\n\r\n    return {\r\n      text: chunk,\r\n      centroid: centroid.toFixed(10), // Limit decimal places for key consistency\r\n      embedding: embeddingArray // You might want to store the array or reconsider how you store the tensor depending on your use case\r\n    };\r\n  }));\r\n  \r\n\r\n  await setupIndexedDB(storeName, embeddings);\r\n}\r\n\r\n\r\n\r\nasync function create_embeddings_open_ai(text, websiteUrl) {\r\n  const token = await (0,_src_js_utils_utils_js__WEBPACK_IMPORTED_MODULE_3__.getToken)();\r\n  // Assuming websiteUrl is sanitized and safe to use as a store name\r\n  const storeName = `embeddings_${websiteUrl.replace(/[^a-zA-Z0-9]/g, '_')}`; // Sanitize URL to create a valid store name\r\n  const splitter = new _splitters_split_js__WEBPACK_IMPORTED_MODULE_4__.CustomTextSplitter(512, 20); // Adjust interval and overlap as needed\r\n  const chunks = splitter.split(text);\r\n  \r\n  const embeddings = await (0,_embeddings_open_ai_embeddings_js__WEBPACK_IMPORTED_MODULE_2__.sendTextForEmbeddings)(token, chunks);\r\n  \r\n  // await Promise.allSettled(chunks.map(async (chunk) => {\r\n  //   console.log(\"me inside the embedder\")\r\n  //   const tensorEmbedding = await embed(chunk);\r\n  const bigDat = chunks.map(async (chunk, i) => {\r\n    \r\n    // Assuming tensorEmbedding.data contains the Float32Array of embedding values\r\n    const embeddingArray = Array.from(embeddings[i]); // Convert Float32Array to regular array\r\n    // Now you can use .reduce on the \r\n    \r\n    \r\n\r\n    let centroid = embeddingArray.reduce((acc, val) => acc + val, 0) / embeddingArray.length;\r\n     centroid = !isNaN(numericCentroid) && isFinite(numericCentroid) ? numericCentroid.toFixed(15) : \"fallbackUniqueKey\";\r\n\r\n    return {\r\n      text: chunk,\r\n      centroid: centroid.toFixed(15), // Limit decimal places for key consistency\r\n      embedding: embeddingArray // You might want to store the array or reconsider how you store the tensor depending on your use case\r\n    };\r\n  })\r\n  \r\n\r\n  await setupIndexedDB(storeName, bigDat);\r\n}\r\n\r\n\r\nasync function setupIndexedDB(storeName, embeddings) {  \r\n  const db = await (0,idb__WEBPACK_IMPORTED_MODULE_5__.openDB)('tinyML', 1, {\r\n    upgrade(db, oldVersion, newVersion, transaction) {\r\n      if (!db.objectStoreNames.contains(storeName)) {\r\n        const store = db.createObjectStore(storeName, {\r\n          keyPath: 'centroid'\r\n        });\r\n      }\r\n    },\r\n  });\r\n\r\n  const tx = db.transaction(storeName, 'readwrite');\r\n  embeddings.forEach(({ text, centroid }) => {\r\n    tx.store.put({ centroid, text });\r\n  });\r\n  await tx.done;\r\n\r\n// initializeDatabase('tinyML', storeName).then(() => console.log('Database is ready'));\r\n\r\n}\r\n\r\n\r\n\r\nasync function initializeDatabase(dbName, tableName) {\r\n  await _db_database_js__WEBPACK_IMPORTED_MODULE_1__.databaseIndicesClassSingleton.initInstance(dbName,tableName);\r\n}\r\n\r\n\r\n\n\n//# sourceURL=webpack://injectchat/./frontend_engine/loaders/ini_embeddings.js?");

/***/ }),

/***/ "./frontend_engine/splitters/split.js":
/*!********************************************!*\
  !*** ./frontend_engine/splitters/split.js ***!
  \********************************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   CustomTextSplitter: () => (/* binding */ CustomTextSplitter)\n/* harmony export */ });\nclass CustomTextSplitter {\r\n    // intervals in char, overlap in words...\r\n    constructor(interval, overlap, charToSplit = '\\n\\n') {\r\n        this.interval = interval;\r\n        this.overlap = overlap;\r\n        this.charToSplit = charToSplit;\r\n    }\r\n\r\n    /**\r\n     * Splits the given text into chunks based on a specified interval and overlap.\r\n     *\r\n     * @param {string} text - The text to be split into chunks.\r\n     * @return {Array<string>} An array of strings representing the chunks of text.\r\n     */\r\n    split(text) {\r\n        const result = [];\r\n        const lines = text.split(this.charToSplit);\r\n        let overlapBuffer = [];\r\n\r\n        for (const line of lines) {\r\n            // Split by words and punctuation\r\n            const tokens = line.match(/\\w+|[^\\w\\s]+/g) || [];\r\n            let chunk = overlapBuffer.join(' ') + (overlapBuffer.length > 0 ? ' ' : ''); // Start new chunk with overlap if any\r\n            overlapBuffer = []; // Clear overlap buffer after using it\r\n\r\n            for (const token of tokens) {\r\n                // Calculate new chunk size with current token\r\n                const newChunkSize = chunk.length + token.length + (chunk.length > 0 ? 1 : 0);\r\n                if (newChunkSize > this.interval && chunk.length > 0) {\r\n                    result.push(chunk.trim());\r\n                    chunk = ''; // Reset chunk\r\n\r\n                    // Ensure overlapBuffer is used to start the new chunk\r\n                    if (overlapBuffer.length > 0) {\r\n                        chunk = overlapBuffer.join(' ') + ' ';\r\n                    }\r\n                }\r\n\r\n                // Update chunk and overlap buffer with the current token\r\n                chunk += token + ' ';\r\n                overlapBuffer.push(token);\r\n                \r\n                // Maintain overlap buffer size\r\n                if (overlapBuffer.length > this.overlap) {\r\n                    overlapBuffer = overlapBuffer.slice(-this.overlap); // Keep only the last 'overlap' number of words\r\n                }\r\n            }\r\n\r\n            // Add the last chunk if it has content\r\n            if (chunk.trim().length > 0) {\r\n                result.push(chunk.trim());\r\n            }\r\n        }\r\n        return result;\r\n    }\r\n}\n\n//# sourceURL=webpack://injectchat/./frontend_engine/splitters/split.js?");

/***/ }),

/***/ "./src/js/utils/utils.js":
/*!*******************************!*\
  !*** ./src/js/utils/utils.js ***!
  \*******************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   alert: () => (/* binding */ alert),\n/* harmony export */   debounce: () => (/* binding */ debounce),\n/* harmony export */   getToken: () => (/* binding */ getToken),\n/* harmony export */   saveToken: () => (/* binding */ saveToken)\n/* harmony export */ });\n/* harmony import */ var idb__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! idb */ \"./node_modules/idb/build/index.js\");\n // Ensure you have `idb` library installed or included\r\n\r\n\r\nfunction debounce(func, wait) {\r\n  let timeout;\r\n  return function(...args) {\r\n    const context = this;\r\n    clearTimeout(timeout);\r\n    timeout = setTimeout(() => func.apply(context, args), wait);\r\n  };\r\n}\r\n\r\n\r\nfunction alert(text, type){\r\nconst token = 'none'\r\n  if (type===\"success\") {\r\n    // Show the success notice\r\n    const successNotice = document.getElementById(\"success-notice\");\r\n    successNotice.style.display = \"block\";\r\n    successNotice.textContent = text\r\n\r\n    // Hide the success notice after 3 seconds\r\n    setTimeout(() => {\r\n      successNotice.style.display = \"none\";\r\n    }, 3000);\r\n    return token;\r\n  } else {\r\n    console.error(text);\r\n    const errorNotice = document.getElementById(\"error-notice\");\r\n    errorNotice.style.display = \"block\";\r\n    errorNotice.textContent = text;\r\n\r\n    // Hide the error notice after 3 seconds\r\n    setTimeout(() => {\r\n      errorNotice.style.display = \"none\";\r\n    }, 3000);\r\n  }\r\n}\r\nasync function saveToken(token) {\r\n  let db;\r\n  try {\r\n    db = await (0,idb__WEBPACK_IMPORTED_MODULE_0__.openDB)('tinyML_tokenS', 1, {\r\n      upgrade(db) {\r\n        if (!db.objectStoreNames.contains('tokens')) {\r\n          db.createObjectStore('tokens', { keyPath: 'id' });\r\n        }\r\n      },\r\n    });\r\n    const tx = db.transaction('tokens', 'readwrite');\r\n    const store = tx.objectStore('tokens');\r\n    await store.put({ id: \"token_avaya\", value: token });\r\n    await tx.done; // Ensures the transaction completes successfully\r\n  } catch (error) {\r\n    console.error(\"Failed to save token:\", error);\r\n    // Handle the error (e.g., by retrying, logging, or notifying the user)\r\n  } finally {\r\n    if (db) db.close(); // Close the database connection to prevent leaks\r\n  }\r\n}\r\n\r\n\r\nasync function getToken() {\r\n  const db = await (0,idb__WEBPACK_IMPORTED_MODULE_0__.openDB)('tinyML_token', 1);\r\n  const tx = db.transaction('tokens', 'readonly');\r\n  const store = tx.objectStore('tokens');\r\n  const tokenObject = await store.get(\"token_avaya\");\r\n  await tx.done;\r\n  return tokenObject ? tokenObject.value : null;\r\n}\r\n\n\n//# sourceURL=webpack://injectchat/./src/js/utils/utils.js?");

/***/ }),

/***/ "?2ca1":
/*!**********************************!*\
  !*** onnxruntime-node (ignored) ***!
  \**********************************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/onnxruntime-node_(ignored)?");

/***/ }),

/***/ "?0a40":
/*!********************!*\
  !*** fs (ignored) ***!
  \********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/fs_(ignored)?");

/***/ }),

/***/ "?61c2":
/*!**********************!*\
  !*** path (ignored) ***!
  \**********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/path_(ignored)?");

/***/ }),

/***/ "?0740":
/*!***********************!*\
  !*** sharp (ignored) ***!
  \***********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/sharp_(ignored)?");

/***/ }),

/***/ "?66bb":
/*!****************************!*\
  !*** stream/web (ignored) ***!
  \****************************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/stream/web_(ignored)?");

/***/ }),

/***/ "?0a9a":
/*!********************!*\
  !*** fs (ignored) ***!
  \********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/fs_(ignored)?");

/***/ }),

/***/ "?73ea":
/*!**********************!*\
  !*** path (ignored) ***!
  \**********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/path_(ignored)?");

/***/ }),

/***/ "?845f":
/*!*********************!*\
  !*** url (ignored) ***!
  \*********************/
/***/ (() => {

eval("/* (ignored) */\n\n//# sourceURL=webpack://injectchat/url_(ignored)?");

/***/ })

}]);
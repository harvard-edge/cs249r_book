// import { SortedArray } from "./utils/SortedArray_lodash";


//  const DB_DEFAULTS = {
//   dbName: "vectorDB",
//   objectStore: "vectors",
//   hyperplanes: 10, // Number of dimensions for hashing
//   dimensions: 384	,  // Dimension of the vectors
//   numPlanes: 5     // Number of hyperplanes for LSH
// };

let DB_DEFAULTS = {
  dbName: "vectorDB_new",
  objectStore: "vectors",
  hyperplanes: 10,
  dimensions: 384,
  numPlanes: 5
};

// Function to update defaults
export function setDBDefaults(newValues) {
  DB_DEFAULTS = { ...DB_DEFAULTS, ...newValues };
}

// Function to get the current defaults
export function getDBDefaults() {
  return DB_DEFAULTS;
}

function generateRandomVector(dimensions) {
  return Array.from({length: dimensions}, () => Math.random() - 0.5);
}

class LSH {
  constructor(dimensions, numPlanes, numTables = 5) {
      this.numTables = numTables;
      this.tables = Array.from({ length: numTables }, () =>
          Array.from({ length: numPlanes }, () => generateRandomVector(dimensions))
      );
  }

  hashVector(vector) {
      return this.tables.map(planes =>
          planes.map(plane =>
              vector.reduce((acc, v, idx) => acc + v * plane[idx], 0) >= 0 ? '1' : '0'
          ).join('')
      );
  }
}


function cosineSimilarity(a, b) {

  // temp workaround
  if(a.length === 1){
    a = a[0]
  }
  
  const dotProduct = a.reduce((sum, aVal, idx) => sum + aVal * b[idx], 0);
  const aMagnitude = Math.sqrt(a.reduce((sum, aVal) => sum + aVal * aVal, 0));
  const bMagnitude = Math.sqrt(b.reduce((sum, bVal) => sum + bVal * bVal, 0));

  return dotProduct / (aMagnitude * bMagnitude);
}

function create(options) {
  const { dbName, objectStore, vectorPath } = {
    ...DB_DEFAULTS,
    ...options,
  };


  return new Promise((resolve, reject) => {
    const request = indexedDB.open(dbName, 2); // Ensure a version number that triggers onupgradeneeded

    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(objectStore)) {
        db.createObjectStore(objectStore, { autoIncrement: true });
      }
      const hashIndexStoreName = `${objectStore}_hashIndex`;
      if (!db.objectStoreNames.contains(hashIndexStoreName)) {
        db.createObjectStore(hashIndexStoreName, { autoIncrement: true });
      }
    };

    request.onsuccess = (event) => {
      resolve(event.target.result);
    };

    request.onerror = (event) => {
      reject(event.target.error);
    };
  });
}

class VectorDB {
  #objectStore;
  #vectorPath;
  #db;
  #lsh;

  // constructor(options) {
  //   const { dbName, objectStore, vectorPath, dimensions, numPlanes } = {
  //     ...DB_DEFAULTS,
  //     ...options,
  //   };

  constructor(options = {}) {
    const { dbName, objectStore, vectorPath, dimensions, numPlanes } = {
      ...DB_DEFAULTS,
      ...options,
    };

    this.#objectStore = objectStore;
    this.#vectorPath = vectorPath;
    this.#lsh = new LSH(dimensions, numPlanes);
    this.#db = create({dbName, objectStore, vectorPath});
  }


async insert(object) {
  const vector = object[this.#vectorPath];
  if (!Array.isArray(vector) && !(vector instanceof Int8Array)) {
      throw new Error(`${this.#vectorPath} on 'object' is expected to be an Array or Int8Array`);
  }

  const db = await this.#db;
  const transaction = db.transaction([this.#objectStore, `${this.#objectStore}_hashIndex`], "readwrite");
  const store = transaction.objectStore(this.#objectStore);
  const hashIndexStore = transaction.objectStore(`${this.#objectStore}_hashIndex`);

  try {
      const request = store.add(object);
      const result = await new Promise((resolve, reject) => {
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
      });

      // Compute hashes for all tables
      const hashes = this.#lsh.hashVector(vector);
      for (let hash of hashes) {
          const bucket = await new Promise((resolve, reject) => {
              const indexRequest = hashIndexStore.get(hash);
              indexRequest.onsuccess = () => resolve(indexRequest.result || []);
              indexRequest.onerror = () => reject(indexRequest.error);
          });

          // Add the new vector key to the bucket
          bucket.push(result);
          await new Promise((resolve, reject) => {
              const putRequest = hashIndexStore.put(bucket, hash);
              putRequest.onsuccess = () => resolve();
              putRequest.onerror = () => reject(putRequest.error);
          });
      }

      return result;
  } catch (error) {
      console.error('Database error during insertion:', error);
      throw error; // Re-throw to signal error condition to caller
  }
}



async delete(key) {
  if (key == null) {
      throw new Error("Unable to delete object without a key");
  }

  const db = await this.#db;
  const transaction = db.transaction([this.#objectStore, `${this.#objectStore}_hashIndex`], "readwrite");
  const store = transaction.objectStore(this.#objectStore);
  const hashIndexStore = transaction.objectStore(`${this.#objectStore}_hashIndex`);

  const object = await new Promise((resolve, reject) => {
      const request = store.get(key);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
  });

  if (!object) {
      throw new Error("Object not found with the provided key");
  }

  const vector = object[this.#vectorPath];
  const hashKeys = this.#lsh.hashVector(vector);

  // Remove the key from all hash buckets across multiple tables
  await Promise.all(hashKeys.map(hashKey => this.removeFromBucket(hashIndexStore, key, hashKey)));

  // Finally, delete the object from the main store
  return new Promise((resolve, reject) => {
      const deleteRequest = store.delete(key);
      deleteRequest.onsuccess = () => resolve();
      deleteRequest.onerror = () => reject(deleteRequest.error);
  });
}

async removeFromBucket(hashIndexStore, key, hashKey) {
  const bucket = await new Promise((resolve, reject) => {
      const request = hashIndexStore.get(hashKey);
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
  });

  const index = bucket.indexOf(key);
  if (index !== -1) {
      bucket.splice(index, 1);
      await new Promise((resolve, reject) => {
          const request = hashIndexStore.put(bucket, hashKey);
          request.onsuccess = () => resolve();
          request.onerror = () => reject(request.error);
      });
  }
}



async update(key, object) {
  if (key == null) {
      throw new Error("Unable to update object without a key");
  }

  if (!(this.#vectorPath in object)) {
      throw new Error(`${this.#vectorPath} expected to be present in the object being updated`);
  }

  if (!Array.isArray(object[this.#vectorPath]) && !(object[this.#vectorPath] instanceof Int8Array)) {
      throw new Error(`${this.#vectorPath} on 'object' is expected to be an Array or Int8Array`);
  }

  const db = await this.#db;
  const transaction = db.transaction([this.#objectStore, `${this.#objectStore}_hashIndex`], "readwrite");
  const store = transaction.objectStore(this.#objectStore);
  const hashIndexStore = transaction.objectStore(`${this.#objectStore}_hashIndex`);

  const currentObjectRequest = store.get(key);

  return new Promise((resolve, reject) => {
      currentObjectRequest.onsuccess = async () => {
          const currentObject = currentObjectRequest.result;
          if (!currentObject) {
              reject(new Error("Object not found with the provided key"));
              return;
          }
          const oldHashes = this.#lsh.hashVector(currentObject[this.#vectorPath]);
          const newHashes = this.#lsh.hashVector(object[this.#vectorPath]);

          const updateRequest = store.put(object, key);
          updateRequest.onsuccess = async () => {
              try {
                  for (let i = 0; i < this.#lsh.numTables; i++) {
                      if (oldHashes[i] !== newHashes[i]) {
                          await this.updateHashIndex(hashIndexStore, key, oldHashes[i], newHashes[i]);
                      }
                  }
                  resolve(updateRequest.result); // Resolve with the update result
              } catch (error) {
                  reject(error);
              }
          };
          updateRequest.onerror = () => reject(updateRequest.error);
      };
      currentObjectRequest.onerror = () => reject(currentObjectRequest.error);
  });
}

async updateHashIndex(hashIndexStore, key, oldHash, newHash) {
  const oldBucketRequest = hashIndexStore.get(oldHash);
  const oldBucket = await new Promise((resolve, reject) => {
      oldBucketRequest.onsuccess = () => resolve(oldBucketRequest.result || []);
      oldBucketRequest.onerror = () => reject(oldBucketRequest.error);
  });

  const index = oldBucket.indexOf(key);
  if (index !== -1) {
      oldBucket.splice(index, 1);
      await hashIndexStore.put(oldBucket, oldHash);
  }

  const newBucketRequest = hashIndexStore.get(newHash);
  const newBucket = await new Promise((resolve, reject) => {
      newBucketRequest.onsuccess = () => resolve(newBucketRequest.result || []);
      newBucketRequest.onerror = () => reject(newBucketRequest.error);
  });

  newBucket.push(key);
  await hashIndexStore.put(newBucket, newHash);
}



async query(queryVector, options = { limit: 10 }) {
  // console.log("Querying for vector:", queryVector, options);
  const { limit } = options;
  let collectedKeys = new Set();
  let resultObjects = [];

  try {
      const db = await this.#db;
      const transaction = db.transaction([this.#objectStore, `${this.#objectStore}_hashIndex`], "readonly");
      const store = transaction.objectStore(this.#objectStore);
      const hashIndexStore = transaction.objectStore(`${this.#objectStore}_hashIndex`);

      // Compute hashes for all tables
      const hashes = this.#lsh.hashVector(queryVector);
      for (let hash of hashes) {
          const bucket = await new Promise((resolve, reject) => {
              const indexRequest = hashIndexStore.get(hash);
              indexRequest.onsuccess = () => resolve(indexRequest.result || []);
              indexRequest.onerror = () => reject(indexRequest.error);
          });

          // Process each key in the bucket if not already processed
          for (let key of bucket) {
              if (!collectedKeys.has(key)) {
                  collectedKeys.add(key);
                  const vector = await new Promise((resolve, reject) => {
                      const vectorRequest = store.get(key);
                      vectorRequest.onsuccess = () => resolve(vectorRequest.result);
                      vectorRequest.onerror = () => reject(vectorRequest.error);
                  });
                  const similarity = cosineSimilarity(queryVector, vector[this.#vectorPath]);
                  resultObjects.push({ object: vector, key, similarity });
              }
          }
      }

      // Sort and limit results
      resultObjects.sort((a, b) => b.similarity - a.similarity); // Descending order by similarity
      return resultObjects.slice(0, limit);
  } catch (error) {
      console.error("Error during query operation:", error);
      throw error; // Rethrow or handle error as needed
  }
}


  get objectStore() {
    // Escape hatch.
    return this.#objectStore;
  }
}

export { VectorDB };
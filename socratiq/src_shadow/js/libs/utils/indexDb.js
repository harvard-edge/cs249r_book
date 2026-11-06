// src_shadow/js/libs/utils/indexdbUtils.js

export class IndexDBManager {
    constructor(dbName, version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
    }

    async getCurrentVersion() {
        return new Promise((resolve) => {
            const request = indexedDB.open(this.dbName);
            request.onsuccess = (event) => {
                const db = event.target.result;
                const version = db.version;
                db.close();
                resolve(version);
            };
            request.onerror = () => {
                console.error('Error getting current version');
                resolve(1);
            };
        });
    }

    async initDB(stores) {
        this._lastKnownStores = stores;
        if (!this._initPromise) {
            this._initPromise = this._openWithSchema(stores).finally(() => {
                this._initPromise = null;
            });
        }
        return this._initPromise;
    }

    async _openWithSchema(stores) {
        // Inspect the installed schema before choosing an upgrade version. A newer
        // database may belong to another SocratiQ release and must never be reset.
        if (this.db) {
            this.db.close();
            this.db = null;
        }
        const open = (version) => new Promise((resolve, reject) => {
            const request = version === undefined
                ? indexedDB.open(this.dbName)
                : indexedDB.open(this.dbName, version);
            request.onerror = () => reject(request.error);
            request.onupgradeneeded = () => {
                const db = request.result;
                for (const { name, keyPath, indexes = [] } of stores) {
                    const store = db.objectStoreNames.contains(name)
                        ? request.transaction.objectStore(name)
                        : db.createObjectStore(name, { keyPath });
                    for (const { name: indexName, keyPath: indexKeyPath, unique, multiEntry } of indexes) {
                        if (!store.indexNames.contains(indexName)) {
                            store.createIndex(indexName, indexKeyPath, { unique, multiEntry });
                        }
                    }
                }
            };
            request.onsuccess = () => resolve(request.result);
        });
        let db = await open();
        const missingSchema = stores.some(({ name, indexes = [] }) => {
            if (!db.objectStoreNames.contains(name)) return true;
            const store = db.transaction(name, 'readonly').objectStore(name);
            return indexes.some(index => !store.indexNames.contains(index.name));
        });
        const targetVersion = Math.max(this.version, db.version + (missingSchema ? 1 : 0));
        if (targetVersion > db.version) {
            db.close();
            db = await open(targetVersion);
        }
        this.version = db.version;
        this.db = db;
        db.onversionchange = () => {
            db.close();
            if (this.db === db) this.db = null;
        };
        db.onclose = () => {
            if (this.db === db) this.db = null;
        };
        return db;
    }

    async add(storeName, data) {
        return this.performTransaction(storeName, 'readwrite', (store) => {
            return store.add(data);
        });
    }

    async update(storeName, data) {

        // Ensure data has the correct keyPath for the store
        if (storeName === 'chapterSummaries' && !data.chapterId) {
            throw new Error('Missing required chapterId for chapterSummaries store');
        }
        
        
        return this.performTransaction(storeName, 'readwrite', (store) => {
            return store.put(data);
        });
    }

    // Add put method for compatibility with TOC extractor
    async put(storeName, data) {
        return this.performTransaction(storeName, 'readwrite', (store) => {
            return store.put(data);
        });
    }

    // Add get method for compatibility with TOC extractor
    async get(storeName, key) {
        return this.performTransaction(storeName, 'readonly', (store) => {
            return store.get(key);
        });
    }

    async delete(storeName, key) {
        return this.performTransaction(storeName, 'readwrite', (store) => {
            return store.delete(key);
        });
    }

    async getAll(storeName) {
        return this.performTransaction(storeName, 'readonly', (store) => {
            return store.getAll();
        });
    }

    async getByKey(storeName, key) {
        return this.performTransaction(storeName, 'readonly', (store) => {
            return store.get(key);
        });
    }

    async getLatest(storeName) {
        
        return new Promise((resolve, reject) => {
            if (!this.db) {
                console.error("Database not initialized");
                reject(new Error("Database not initialized"));
                return;
            }

            const transaction = this.db.transaction(storeName, 'readonly');
            const store = transaction.objectStore(storeName);

            if (!store.indexNames.contains('date')) {
                console.error("Date index not found");
                reject(new Error("Date index not found in store"));
                return;
            }

            const index = store.index('date');
            
            const request = index.openCursor(null, 'prev');
            
            request.onerror = (event) => {
                console.error("Error in cursor request:", event.target.error);
                reject(request.error);
            };
            
            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    resolve(cursor.value);
                } else {
                    resolve(null);
                }
            };

            transaction.oncomplete = () => {
            };

            transaction.onerror = (event) => {
                console.error("Transaction failed:", event.target.error);
                reject(event.target.error);
            };
        });
    }

    async performTransaction(storeName, mode, operation) {
        for (let attempt = 0; attempt < 2; attempt++) {
            try {
                if (!this.db) await this.reconnectDB();
                return await new Promise((resolve, reject) => {
                    const transaction = this.db.transaction(storeName, mode);
                    let result;
                    const timeoutId = setTimeout(() => {
                        try { transaction.abort(); } catch {}
                        reject(new Error('Transaction timeout'));
                    }, 5000);
                    const fail = error => {
                        clearTimeout(timeoutId);
                        reject(error || new Error('Database transaction aborted'));
                    };
                    transaction.oncomplete = () => {
                        clearTimeout(timeoutId);
                        resolve(result);
                    };
                    transaction.onabort = () => fail(transaction.error);
                    transaction.onerror = () => fail(transaction.error);
                    try {
                        const request = operation(transaction.objectStore(storeName));
                        request.onsuccess = () => { result = request.result; };
                        request.onerror = () => fail(request.error);
                    } catch (error) {
                        try { transaction.abort(); } catch {}
                        fail(error);
                    }
                });
            } catch (error) {
                if (error.name !== 'InvalidStateError' || attempt > 0) throw error;
                this.db = null;
            }
        }
    }

    async reconnectDB() {
        await this.initDB(this._lastKnownStores || []);
        return true;
    }
}

let dbInstances = { socratiqDB: null };
let initializationPromise = null;

export async function initializeSocratiqDB() {
    if (!initializationPromise) {
        initializationPromise = (async () => {
            const { DB_CONFIGS } = await import('../../../configs/db_configs_one.js');
            const manager = dbInstances.socratiqDB || new IndexDBManager(DB_CONFIGS.name, DB_CONFIGS.version);
            await manager.initDB(DB_CONFIGS.stores);
            dbInstances.socratiqDB = manager;
            return manager;
        })().finally(() => { initializationPromise = null; });
    }
    return initializationPromise;
}

// Update the getter to always return the socratiqDB instance
export async function getDBInstance(dbName = 'socratiqDB') {
    if (!dbInstances.socratiqDB) {
        console.warn('Database instance not found, initializing...');
        return await initializeSocratiqDB();
    }
    if (!dbInstances.socratiqDB.db) await dbInstances.socratiqDB.reconnectDB();
    return dbInstances.socratiqDB;
}

// Update the setter to properly store the instance
export function setDBInstance(instance, dbName = 'socratiqDB') {
    dbInstances.socratiqDB = instance;
    return instance;
}

// Add a utility function to check if stores exist
export async function verifyStores() {
    const dbManager = await getDBInstance();
    if (dbManager && dbManager.db) {
        const storeNames = Array.from(dbManager.db.objectStoreNames);
        return storeNames;
    }
    return [];
}

// Add this new function
async function deleteDatabase(dbName) {
    return new Promise((resolve, reject) => {
        // Close any existing connections
        if (dbInstances.socratiqDB && dbInstances.socratiqDB.db) {
            dbInstances.socratiqDB.db.close();
        }
        
        const request = indexedDB.deleteDatabase(dbName);
        
        request.onsuccess = () => {
            resolve(true);
        };
        
        request.onerror = (event) => {
            console.error(`Error deleting database ${dbName}:`, event.target.error);
            reject(event.target.error);
        };
        
        request.onblocked = (event) => {
            console.warn(`Database deletion blocked, possibly due to open connections`);
            // Try to close any remaining connections
            if (dbInstances.socratiqDB && dbInstances.socratiqDB.db) {
                dbInstances.socratiqDB.db.close();
            }
        };
    });
}

// Legacy startup callers use this name for schema repair. Repair in place so
// stores belonging to older or newer releases remain intact, including unknown stores.
export async function forceInitializeSocratiqDB() {
    return initializeSocratiqDB();
}

// Add this new function to handle complete reset
export async function resetSocratiqDB() {
    try {
        const { DB_CONFIGS } = await import('../../../configs/db_configs_one.js');
        
        // First, delete the existing database
        await deleteDatabase(DB_CONFIGS.name);
        
        // Clear any existing instances
        dbInstances = {
            socratiqDB: null
        };
        
        // Initialize a fresh database
        const newDb = await initializeSocratiqDB();
        
        // Create default stores with fresh configuration
        await newDb.initDB(DB_CONFIGS.stores);
        
        return true;
        
    } catch (error) {
        console.error('Failed to reset SocratiQ database:', error);
        throw error;
    }
}
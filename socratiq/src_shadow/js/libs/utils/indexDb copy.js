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
        console.log(`Initializing DB: ${this.dbName}`);
        this._lastKnownStores = stores;
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = (event) => {
                console.error(`Database error for ${this.dbName}:`, event.target.error);
                reject(event.target.error);
            };

            request.onsuccess = (event) => {
                console.log(`Database ${this.dbName} opened successfully`);
                this.db = event.target.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                console.log(`Upgrading database ${this.dbName} to version ${this.version}`);
                const db = event.target.result;
                
                // Only create/update stores if this is a version upgrade
                if (event.oldVersion < event.newVersion) {
                    stores.forEach(({ name, keyPath, indexes }) => {
                        if (!db.objectStoreNames.contains(name)) {
                            console.log(`Creating store: ${name}`);
                            const store = db.createObjectStore(name, { keyPath });
                            
                            if (indexes) {
                                indexes.forEach(({ name: indexName, keyPath: indexKeyPath }) => {
                                    if (!store.indexNames.contains(indexName)) {
                                        store.createIndex(indexName, indexKeyPath);
                                    }
                                });
                            }
                        }
                    });
                }
            };
        });
    }

    async add(storeName, data) {
        return this.performTransaction(storeName, 'readwrite', (store) => {
            return store.add(data);
        });
    }

    async update(storeName, data) {

        console.log("update storeName", storeName, data);
        // Ensure data has the correct keyPath for the store
        if (storeName === 'chapterSummaries' && !data.chapterId) {
            throw new Error('Missing required chapterId for chapterSummaries store');
        }
        
        // Log the data being saved
        console.log('Updating store:', storeName, 'with data:', JSON.stringify(data, null, 2));
        
        return this.performTransaction(storeName, 'readwrite', (store) => {
            console.log("update store", store, "with", data)
            return store.put(data);
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
        console.log("Getting latest report from store:", storeName);
        
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
            console.log("Got date index");
            
            const request = index.openCursor(null, 'prev');
            
            request.onerror = (event) => {
                console.error("Error in cursor request:", event.target.error);
                reject(request.error);
            };
            
            request.onsuccess = (event) => {
                console.log("Cursor request successful");
                const cursor = event.target.result;
                if (cursor) {
                    console.log("Found latest record:", cursor.value);
                    resolve(cursor.value);
                } else {
                    console.log("No records found");
                    resolve(null);
                }
            };

            transaction.oncomplete = () => {
                console.log("Transaction completed successfully");
            };

            transaction.onerror = (event) => {
                console.error("Transaction failed:", event.target.error);
                reject(event.target.error);
            };
        });
    }

    async performTransaction(storeName, mode, operation) {
        console.log(`Starting ${mode} transaction on ${storeName}`);
        
        const MAX_RETRIES = 3;
        let attempts = 0;
        
        while (attempts < MAX_RETRIES) {
            try {
                if (!this.db || this.db.closed) {
                    await this.reconnectDB();
                }

                return new Promise((resolve, reject) => {
                    const transaction = this.db.transaction(storeName, mode);
                    const store = transaction.objectStore(storeName);

                    transaction.onerror = (event) => {
                        console.error("Transaction error:", event.target.error);
                        reject(event.target.error);
                    };

                    // Add timeout protection
                    const timeoutId = setTimeout(() => {
                        try {
                            transaction.abort();
                        } catch (e) {
                            console.warn('Could not abort transaction:', e);
                        }
                        reject(new Error('Transaction timeout'));
                    }, 5000); // 5 second timeout

                    transaction.oncomplete = () => {
                        clearTimeout(timeoutId);
                        console.log("Transaction completed successfully");
                    };

                    try {
                        const request = operation(store);
                        
                        request.onsuccess = () => {
                            clearTimeout(timeoutId);
                            console.log("Operation successful");
                            resolve(request.result);
                        };
                        
                        request.onerror = () => {
                            clearTimeout(timeoutId);
                            console.error("Operation error:", request.error);
                            reject(request.error);
                        };
                    } catch (error) {
                        clearTimeout(timeoutId);
                        console.error("Operation failed:", error);
                        reject(error);
                    }
                });
            } catch (error) {
                attempts++;
                console.error(`Attempt ${attempts} failed:`, error);
                
                if (error.name === 'InvalidStateError' && attempts < MAX_RETRIES) {
                    console.log(`Attempting to reconnect, attempt ${attempts + 1} of ${MAX_RETRIES}`);
                    await new Promise(resolve => setTimeout(resolve, 1000 * attempts)); // Exponential backoff
                    continue;
                }
                throw error;
            }
        }
        throw new Error(`Failed to perform transaction after ${MAX_RETRIES} attempts`);
    }

    async reconnectDB() {
        console.log(`Attempting to reconnect to DB: ${this.dbName}`);
        try {
            await this.initDB(this._lastKnownStores || []);
            return true;
        } catch (error) {
            console.error(`Reconnection failed for ${this.dbName}:`, error);
            return false;
        }
    }
}

// Add after the existing DB initialization code (around line 820)
let dbInstances = {
    socratiqDB: null
};

export async function initializeSocratiqDB() {
    try {
        const { DB_CONFIGS } = await import('../../../configs/db_configs_one.js');
        
        // First check if database already exists and get its configuration
        const existingDb = await new Promise((resolve) => {
            const request = indexedDB.open(DB_CONFIGS.name);
            request.onsuccess = (event) => {
                const db = event.target.result;
                const info = {
                    exists: true,
                    version: db.version,
                    stores: Array.from(db.objectStoreNames)
                };
                db.close();
                resolve(info);
            };
            request.onerror = () => resolve({ exists: false });
        });

        // Check if we need to upgrade by comparing store configurations
        const needsUpgrade = existingDb.exists ? 
            !DB_CONFIGS.stores.every(store => 
                existingDb.stores.includes(store.name)
            ) : true;

        const currentVersion = existingDb.exists ? existingDb.version : 0;
        const newVersion = needsUpgrade ? currentVersion + 1 : currentVersion;
        
        console.log('Database status:', {
            exists: existingDb.exists,
            currentVersion,
            newVersion,
            needsUpgrade
        });
        
        // Create database manager with appropriate version
        const dbManager = new IndexDBManager(DB_CONFIGS.name, newVersion);
        
        // Initialize database with stores
        await dbManager.initDB(DB_CONFIGS.stores);
        
        // Store the instance
        dbInstances.socratiqDB = dbManager;
        
        return dbManager;
        
    } catch (error) {
        console.error('Failed to initialize Socratiq database:', error);
        throw error;
    }
}

// Update the getter to always return the socratiqDB instance
export function getDBInstance(dbName = 'socratiqDB') {
    if (!dbInstances.socratiqDB) {
        console.warn('Database instance not found, initializing...');
        return initializeSocratiqDB();
    }
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
        console.log('Current stores in database:', storeNames);
        return storeNames;
    }
    return [];
}

// Add this new function
async function deleteDatabase(dbName) {
    return new Promise((resolve, reject) => {
        const request = indexedDB.deleteDatabase(dbName);
        request.onsuccess = () => {
            console.log(`Database ${dbName} successfully deleted`);
            resolve();
        };
        request.onerror = () => {
            console.error(`Error deleting database ${dbName}`);
            reject();
        };
    });
}

// Add this to your initialization function if you want to force a fresh start
export async function forceInitializeSocratiqDB() {
    try {
        const { DB_CONFIGS } = await import('../../../configs/db_configs_one.js');
        
        // Get existing data before deletion
        const existingDb = await getDBInstance().catch(() => null);
        const existingData = {};
        
        if (existingDb) {
            // Store existing data from each store
            for (const store of DB_CONFIGS.stores) {
                try {
                    existingData[store.name] = await existingDb.getAll(store.name);
                } catch (e) {
                    console.warn(`Could not backup store ${store.name}:`, e);
                }
            }
        }
        
        // Delete and recreate database
        await deleteDatabase(DB_CONFIGS.name);
        const newDb = await initializeSocratiqDB();
        
        // Restore existing data
        for (const [storeName, data] of Object.entries(existingData)) {
            if (data && data.length > 0) {
                const store = newDb.db
                    .transaction([storeName], 'readwrite')
                    .objectStore(storeName);
                    
                for (const item of data) {
                    await store.put(item);
                }
            }
        }
        
        return newDb;
    } catch (error) {
        console.error('Failed to force initialize database:', error);
        throw error;
    }
}
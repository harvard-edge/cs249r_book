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
        
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = (event) => {
                console.error(`❌ Database error for ${this.dbName}:`, event.target.error);
                reject(event.target.error);
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                console.log(`✅ Database ${this.dbName} opened successfully`);
                console.log(`✅ Available stores:`, Array.from(this.db.objectStoreNames));
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                console.log(`🔄 Database upgrade needed for ${this.dbName}`);
                console.log(`🔄 Old version: ${event.oldVersion}, New version: ${event.newVersion}`);
                
                const db = event.target.result;
                
                // Create/update stores
                stores.forEach(({ name, keyPath, indexes }) => {
                    console.log(`🔄 Processing store: ${name}`);
                    
                    if (!db.objectStoreNames.contains(name)) {
                        console.log(`➕ Creating new store: ${name}`);
                        const store = db.createObjectStore(name, { keyPath });
                        
                        if (indexes) {
                            indexes.forEach(({ name: indexName, keyPath: indexKeyPath, unique, multiEntry }) => {
                                if (!store.indexNames.contains(indexName)) {
                                    console.log(`➕ Creating index: ${indexName} on ${indexKeyPath}`);
                                    store.createIndex(indexName, indexKeyPath, { unique, multiEntry });
                                }
                            });
                        }
                    } else {
                        console.log(`✅ Store ${name} already exists`);
                    }
                });
                
                console.log(`✅ Database upgrade completed for ${this.dbName}`);
            };
        });
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
                        console.error("❌ IDB Transaction error:", event.target.error);
                        reject(event.target.error);
                    };

                    // Add timeout protection
                    const timeoutId = setTimeout(() => {
                        console.warn("⏰ IDB Transaction timeout");
                        try {
                            transaction.abort();
                        } catch (e) {
                            console.warn('Could not abort transaction:', e);
                        }
                        reject(new Error('Transaction timeout'));
                    }, 5000);

                    transaction.oncomplete = () => {
                        clearTimeout(timeoutId);
                    };

                    try {
                        const request = operation(store);
                        
                        request.onsuccess = () => {
                            clearTimeout(timeoutId);
                            resolve(request.result);
                        };
                        
                        request.onerror = () => {
                            console.error("💥 IDB Operation error:", request.error);
                            clearTimeout(timeoutId);
                            reject(request.error);
                        };
                    } catch (error) {
                        console.error("🚨 IDB Operation failed:", error);
                        clearTimeout(timeoutId);
                        reject(error);
                    }
                });
            } catch (error) {
                attempts++;
                console.error(`❌ IDB Attempt ${attempts} failed:`, error);
                
                if (error.name === 'InvalidStateError' && attempts < MAX_RETRIES) {
                    const delay = 1000 * attempts;
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
                throw error;
            }
        }
        throw new Error(`Failed to perform transaction after ${MAX_RETRIES} attempts`);
    }

    async reconnectDB() {
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
        
        console.log('🔄 Initializing SocratiqDB with version:', DB_CONFIGS.version);
        console.log('🔄 Required stores:', DB_CONFIGS.stores.map(s => s.name));
        
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
                console.log('📊 Existing database info:', info);
                db.close();
                resolve(info);
            };
            request.onerror = () => {
                console.log('📊 No existing database found');
                resolve({ exists: false });
            };
        });

        // Check if we need to upgrade by comparing store configurations
        const needsUpgrade = existingDb.exists ? 
            !DB_CONFIGS.stores.every(store => 
                existingDb.stores.includes(store.name)
            ) : true;

        const currentVersion = existingDb.exists ? existingDb.version : 0;
        const newVersion = needsUpgrade ? DB_CONFIGS.version : currentVersion;
        
        console.log('🔄 Database upgrade needed:', needsUpgrade);
        console.log('🔄 Current version:', currentVersion, 'New version:', newVersion);
        
        // Create database manager with appropriate version
        const dbManager = new IndexDBManager(DB_CONFIGS.name, newVersion);
        
        // Initialize database with stores
        await dbManager.initDB(DB_CONFIGS.stores);
        
        // Store the instance
        dbInstances.socratiqDB = dbManager;
        
        console.log('✅ SocratiqDB initialized successfully');
        console.log('✅ Available stores:', Array.from(dbManager.db.objectStoreNames));
        
        return dbManager;
        
    } catch (error) {
        console.error('❌ Failed to initialize Socratiq database:', error);
        console.error('❌ Error details:', error.message, error.stack);
        throw error;
    }
}

// Update the getter to always return the socratiqDB instance
export async function getDBInstance(dbName = 'socratiqDB') {
    if (!dbInstances.socratiqDB) {
        console.warn('Database instance not found, initializing...');
        return await initializeSocratiqDB();
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
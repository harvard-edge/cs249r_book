// indexedDB-utils.js

export class IndexedDBManager {
    constructor(dbConfig) {
        this.dbConfig = dbConfig;
        this.db = null;
        this.dbUpgradeNeeded = false;
        this._initPromise = null;
    }

    async init() {
        if (!this._initPromise) {
            this._initPromise = this._initialize();
        }
        return this._initPromise;
    }

    async _initialize() {
        try {
            const currentVersion = await this.getCurrentVersion();
            const targetVersion = Math.max(currentVersion + 1, this.dbConfig.version);
            this.db = await this.openDatabase(targetVersion);
            return this.db;
        } catch (error) {
            console.error('Error initializing database:', error);
            throw error;
        }
    }

    async getCurrentVersion() {
        return new Promise((resolve) => {
            const request = indexedDB.open(this.dbConfig.name);
            
            request.onsuccess = (event) => {
                const db = event.target.result;
                const currentVersion = db.version;
                db.close();
                resolve(currentVersion);
            };
            
            request.onerror = () => {
                resolve(0); // Start from version 0 if database doesn't exist
            };
        });
    }

    async openDatabase(version) {
        return new Promise((resolve, reject) => {
            console.log(`Opening database ${this.dbConfig.name} version ${version}`);
            const request = indexedDB.open(this.dbConfig.name, version);

            request.onerror = () => reject(request.error);

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                console.log('Upgrade needed, creating stores...');
                const db = event.target.result;
                
                // Create or update stores based on config
                Object.entries(this.dbConfig.stores).forEach(([storeName, storeConfig]) => {
                    if (!db.objectStoreNames.contains(storeName)) {
                        console.log(`Creating store: ${storeName}`);
                        const store = db.createObjectStore(storeName, {
                            keyPath: storeConfig.keyPath,
                            autoIncrement: storeConfig.autoIncrement || false
                        });

                        // Create indexes if specified
                        storeConfig.indexes?.forEach(indexConfig => {
                            console.log(`Creating index: ${indexConfig.name} for store ${storeName}`);
                            store.createIndex(
                                indexConfig.name, 
                                indexConfig.keyPath || indexConfig.name, 
                                {
                                    unique: indexConfig.unique || false,
                                    multiEntry: indexConfig.multiEntry || false
                                }
                            );
                        });
                    }
                });
            };
        });
    }

    async getDB() {
        if (!this.db) {
            await this.init();
        }
        return this.db;
    }

    async transaction(storeName, mode, callback) {
        try {
            const db = await this.getDB();
            
            // Check if store exists
            if (!db.objectStoreNames.contains(storeName)) {
                console.log(`Store ${storeName} not found, reinitializing database...`);
                await this.init(); // Reinitialize to create missing stores
            }

            return await new Promise((resolve, reject) => {
                const transaction = db.transaction(storeName, mode);
                const store = transaction.objectStore(storeName);

                let result;
                transaction.oncomplete = () => resolve(result);
                transaction.onerror = () => reject(transaction.error);

                // Allow callback to set result
                result = callback(store);
            });
        } catch (error) {
            console.error(`Transaction error in ${storeName}:`, error);
            throw error;
        }
    }

    async get(storeName, key) {
        return this.transaction(storeName, 'readonly', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.get(key);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }

    async put(storeName, data) {
        return this.transaction(storeName, 'readwrite', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.put(data);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }

    async add(storeName, data) {
        return this.transaction(storeName, 'readwrite', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.add(data);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }

    async getAll(storeName, query = null, count = 0) {
        return this.transaction(storeName, 'readonly', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.getAll(query, count);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }

    async delete(storeName, key) {
        return this.transaction(storeName, 'readwrite', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.delete(key);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }

    async clear(storeName) {
        return this.transaction(storeName, 'readwrite', (store) => {
            return new Promise((resolve, reject) => {
                const request = store.clear();
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
        });
    }

    async getByIndex(storeName, indexName, key) {
        return this.transaction(storeName, 'readonly', (store) => {
            return new Promise((resolve, reject) => {
                const index = store.index(indexName);
                const request = index.get(key);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });
        });
    }
}
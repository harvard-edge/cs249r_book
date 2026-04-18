// quizStorage.js
export class QuizStorage {
    constructor(dbName = 'tinyMLChapters', version = 1) {
      this.dbName = dbName;
      this.version = version;
      this.storeName = 'quizTitles';
    }
  
    async initDB() {
      return new Promise((resolve, reject) => {
        const request = indexedDB.open(this.dbName, this.version);
  
        request.onerror = () => reject(request.error);
        
        request.onupgradeneeded = (event) => {
          const db = event.target.result;
          if (!db.objectStoreNames.contains(this.storeName)) {
            db.createObjectStore(this.storeName, { keyPath: 'url' });
          }
        };
  
        request.onsuccess = () => resolve(request.result);
      });
    }
  
    async saveQuizTitles(url, titles) {
      try {
        const db = await this.initDB();
        return new Promise((resolve, reject) => {
          const transaction = db.transaction([this.storeName], 'readwrite');
          const store = transaction.objectStore(this.storeName);
          
          // Use put instead of add to update if exists
          const request = store.put({
            url,
            titles,
            timestamp: new Date().toISOString()
          });
  
          request.onerror = () => reject(request.error);
          request.onsuccess = () => resolve(request.result);
          
          // Close the database connection when the transaction is complete
          transaction.oncomplete = () => db.close();
        });
      } catch (error) {
        console.error('Error saving quiz titles:', error);
        throw error;
      }
    }
  
    async getQuizTitles(url) {
      try {
        const db = await this.initDB();
        return new Promise((resolve, reject) => {
          const transaction = db.transaction([this.storeName], 'readonly');
          const store = transaction.objectStore(this.storeName);
          const request = store.get(url);
  
          request.onerror = () => reject(request.error);
          request.onsuccess = () => {
            resolve(request.result?.titles || null);
            db.close();
          };
        });
      } catch (error) {
        console.error('Error getting quiz titles:', error);
        throw error;
      }
    }
  }
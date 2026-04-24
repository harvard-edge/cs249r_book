// quizStorage.js
import { getDBInstance } from '../utils/indexDb.js';

export class QuizStorage {
    constructor() {
        this.storeName = 'quizTitles';
        this.dbManager = null;
    }
  
    async initDB() {
        if (!this.dbManager) {
            this.dbManager = await getDBInstance();
        }
        return this.dbManager;
    }
  
    async saveQuizTitles(url, titles) {
        try {
            const dbManager = await this.initDB();
            const data = {
                url,
                titles,
                timestamp: new Date().toISOString()
            };
            
            return await dbManager.update(this.storeName, data);
        } catch (error) {
            console.error('Error saving quiz titles:', error);
            throw error;
        }
    }
  
    async getQuizTitles(url) {
        try {
            const dbManager = await this.initDB();
            const result = await dbManager.getByKey(this.storeName, url);
            return result?.titles || null;
        } catch (error) {
            console.error('Error getting quiz titles:', error);
            throw error;
        }
    }
}
let initiate, quickStart, search, deleteRow, update; // Declare variables to hold imported functions
// const callbacks = new Map(); // To track callbacks associated with each action
// let messageId = 0; // A simple counter to ensure uniqueness

self.onmessage = async (event) => {
    // Import the functions once if they haven't been imported yet
    if (!initiate || !quickStart || !quickStart_single || !search || !deleteRow || !update) {
        const module = await import('../vector_db/vectordb/quick_start');
        initiate = module.initiate;
        quickStart = module.quickStart;
        quickStart_single = module.quickStart_single;
        search = module.search;
        deleteRow = module.delete_row;
        update = module.update;

    }

    try {
        let result;

        const { token, ...dataWithoutToken } = event.data;

        switch (event.data.command) {
            case "initiate":
                result = await initiate(event.data.dbName);
                result = {result: result};
                break;
            case 'create':
                result = await quickStart(dataWithoutToken, token, event.data.useLocal);
                result = {result: result, id: event.data.id};
                break;
            case 'create_single':
                    result = await quickStart_single(event.data, token, event.data.useLocal);
                    result = {result: result, id: event.data.id};
                    break;
            case 'search':
                result = await search(dataWithoutToken.text, event.data.k, token, event.data.useLocal);
                result = {result: result, id: event.data.id};

                break;
            case 'delete_row':
                result = await deleteRow(event.data.key);
                result = {result: result, id: event.data.id};

                break;
            case 'update':
                result = await update(event.data.key, dataWithoutToken, token);
                result = {result: result, id: event.data.id};

                break;
            default:
                throw new Error('Unknown command');
        }

        // Post the result back to the main thread
        self.postMessage({ status: 'success', data: result.result, id:result.id });
    } catch (error) {
        self.postMessage({ status: 'error', message: error.message });
    }
};

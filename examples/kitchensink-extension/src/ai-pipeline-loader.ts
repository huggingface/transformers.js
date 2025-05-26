/**
 * AIPipelineLoader - Handles loading and management of AI pipelines
 */

import { pipeline } from '@xenova/transformers';
import type { TaskType, AIManagerState } from './types';
import { AIPipelineConfig } from './ai-pipeline-config';

export class AIPipelineLoader {
    private state: AIManagerState;
    private config: AIPipelineConfig;

    constructor(state: AIManagerState, config: AIPipelineConfig) {
        this.state = state;
        this.config = config;
    }

    public async loadPipeline(taskType: TaskType): Promise<any> {
        const pipelineConfig = this.config.getConfig(taskType);
        if (!pipelineConfig) {
            throw new Error(`No configuration found for task: ${taskType}`);
        }

        const existingPipeline = this.state.pipelines.get(taskType);
        if (existingPipeline) {
            return existingPipeline;
        }

        console.log(`📥 Loading ${taskType} pipeline...`);
        this.state.status.set(taskType, 'loading');
        this.config.updateStatus(taskType, 'loading');

        try {
            const pipelineInstance = await pipeline(
                pipelineConfig.task,
                pipelineConfig.model,
                { quantized: pipelineConfig.quantized }
            );

            this.state.pipelines.set(taskType, pipelineInstance);
            this.state.status.set(taskType, 'loaded');
            this.config.updateStatus(taskType, 'loaded');

            console.log(`✅ ${taskType} pipeline loaded successfully`);
            return pipelineInstance;

        } catch (error) {
            console.error(`❌ Failed to load ${taskType} pipeline:`, error);
            this.state.status.set(taskType, 'error');
            this.config.updateStatus(taskType, 'error');
            throw error;
        }
    }

    public async loadCoreModels(): Promise<void> {
        const coreTypes = this.config.getCoreTaskTypes();
        for (const taskType of coreTypes) {
            try {
                await this.loadPipeline(taskType);
            } catch (error) {
                console.warn(`Failed to load core model ${taskType}:`, error);
            }
        }
    }

    public async loadSecondaryModels(): Promise<void> {
        const secondaryTypes = this.config.getSecondaryTaskTypes();
        for (const taskType of secondaryTypes) {
            try {
                await this.loadPipeline(taskType);
                // Add delay between loads to prevent overwhelming the system
                await new Promise(resolve => setTimeout(resolve, 1000));
            } catch (error) {
                console.warn(`Failed to load secondary model ${taskType}:`, error);
            }
        }
    }

    public getPipeline(taskType: TaskType): any {
        return this.state.pipelines.get(taskType);
    }

    public getStatus(): Record<string, string> {
        const status: Record<string, string> = {};
        for (const [taskType, pipelineStatus] of this.state.status) {
            status[taskType] = pipelineStatus;
        }
        return status;
    }
}

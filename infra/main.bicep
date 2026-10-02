// Scheduled paper-trading job: a Container Apps Job that runs `aihf paper tick`
// on a cron, with the fund ledger on Azure Files so NAV survives restarts.
//
// Nothing here contains a secret. The Financial Datasets key is created empty
// in Key Vault and set out-of-band (see the output at the bottom); the Azure
// OpenAI credential is a managed identity, so there is no key at all.

targetScope = 'resourceGroup'

// Capped at 9 so the derived storage and vault names stay inside their 24-char
// limits once the uniqueness suffix is appended.
@description('Short name used to derive resource names. Lowercase letters and digits.')
@minLength(3)
@maxLength(9)
param appName string = 'aihf'

@description('Location for all resources.')
param location string = resourceGroup().location

@description('Paper fund to advance. Must already exist on the file share.')
param fundName string

@description('Cron for the tick, in UTC. Default 23:30 UTC Mon-Fri = 18:30 EST / 19:30 EDT, after the 16:00 ET close in both.')
param cronExpression string = '30 23 * * 1-5'

@description('Image to run. azd overrides this; the placeholder only lets the template validate.')
param containerImage string = 'mcr.microsoft.com/k8se/quickstart-jobs:latest'

@description('Model id passed to --model. Must be azure/<deployment> to use the managed identity.')
param llmModel string = 'azure/gpt-4o'

@description('Endpoint of an existing Azure OpenAI resource, e.g. https://my-aoai.openai.azure.com')
param azureOpenAiEndpoint string

@description('Name of that Azure OpenAI account, for the role assignment. Empty skips it (grant the role yourself).')
param azureOpenAiAccountName string = ''

var suffix = uniqueString(resourceGroup().id)
// Storage allows no hyphens and caps at 24; Key Vault caps at 24 too. Both
// suffixes are truncated to fit the worst-case appName.
var storageName = toLower('${appName}st${substring(suffix, 0, 10)}')
var vaultName = '${appName}-kv-${substring(suffix, 0, 8)}'
var shareName = 'hedge-fund'
var secretName = 'financial-datasets-api-key'

// Built-in role definition ids.
var acrPullRole = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var kvSecretsUserRole = '4633458b-17de-408a-b874-0445c86b69e6'
var openAiUserRole = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'

resource uami 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${appName}-identity'
  location: location
}

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${appName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    // A paper fund ticks a few times a week; 30 days is plenty to debug one.
    retentionInDays: 30
  }
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-11-01-preview' = {
  name: toLower('${appName}acr${suffix}')
  location: location
  sku: { name: 'Basic' }
  properties: {
    // Managed identity + AcrPull instead; admin keys are a shared secret.
    adminUserEnabled: false
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }

  resource fileServices 'fileServices@2023-05-01' = {
    name: 'default'

    resource share 'shares@2023-05-01' = {
      name: shareName
      properties: {
        // The ledger is small text; this is the floor, not a capacity estimate.
        shareQuota: 100
      }
    }
  }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    // RBAC over access policies: the job's identity gets one scoped role.
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

// Created with no value on purpose: a real key must never enter this template
// or its parameter file. Set it with `az keyvault secret set` after deploy.
resource dataKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: secretName
  properties: {
    value: 'placeholder-set-me-out-of-band'
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${appName}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }

  resource envStorage 'storages@2024-03-01' = {
    name: 'hedgefunddata'
    properties: {
      azureFile: {
        accountName: storage.name
        accountKey: storage.listKeys().keys[0].value
        shareName: shareName
        // The job writes the ledger; read-only would break the whole point.
        accessMode: 'ReadWrite'
      }
    }
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, uami.id, acrPullRole)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRole)
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource kvSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(vault.id, uami.id, kvSecretsUserRole)
  scope: vault
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', kvSecretsUserRole)
    principalId: uami.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Only when the Azure OpenAI account is in this resource group. Cross-group or
// cross-subscription resources need the role granted separately.
module openAiRole 'openai-role.bicep' = if (!empty(azureOpenAiAccountName)) {
  name: 'openai-role'
  params: {
    accountName: azureOpenAiAccountName
    principalId: uami.properties.principalId
    roleDefinitionId: openAiUserRole
  }
}

resource job 'Microsoft.App/jobs@2024-03-01' = {
  name: '${appName}-tick'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${uami.id}': {} }
  }
  properties: {
    environmentId: env.id
    configuration: {
      triggerType: 'Schedule'
      scheduleTriggerConfig: {
        cronExpression: cronExpression
        // One replica, one completion: the ledger is a hash chain, and two
        // writers would race to append the same session.
        parallelism: 1
        replicaCompletionCount: 1
      }
      // No retries. A retry after a partial write cannot be assumed
      // idempotent, and a missed session is recoverable by hand; a corrupted
      // chain is not.
      replicaRetryLimit: 0
      replicaTimeout: 3600
      registries: [
        {
          server: acr.properties.loginServer
          identity: uami.id
        }
      ]
      secrets: [
        {
          name: secretName
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${secretName}'
          identity: uami.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'tick'
          image: containerImage
          args: ['--model', llmModel, 'paper', 'tick', fundName]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'FINANCIAL_DATASETS_API_KEY'
              secretRef: secretName
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAiEndpoint
            }
            {
              // DefaultAzureCredential needs this to pick the user-assigned
              // identity; without it the job falls back and fails to auth.
              name: 'AZURE_CLIENT_ID'
              value: uami.properties.clientId
            }
          ]
          volumeMounts: [
            {
              volumeName: 'data'
              // Matches ENV HOME=/data in the Dockerfile, which is what puts
              // ~/.hedge-fund on the share.
              mountPath: '/data'
            }
          ]
        }
      ]
      volumes: [
        {
          name: 'data'
          storageType: 'AzureFile'
          storageName: env::envStorage.name
        }
      ]
    }
  }
  dependsOn: [acrPull, kvSecretsUser]
}

output registryLoginServer string = acr.properties.loginServer
output jobName string = job.name
output storageAccountName string = storage.name
output fileShareName string = shareName
output identityClientId string = uami.properties.clientId

@description('Run this to install the real data key; it must not live in the template.')
output setSecretCommand string = 'az keyvault secret set --vault-name ${vault.name} --name ${secretName} --value <your-financial-datasets-key>'

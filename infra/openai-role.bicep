// Grants the job's identity Cognitive Services OpenAI User on an existing
// Azure OpenAI account. Separate module so the role is scoped to that account
// rather than the whole resource group.

@description('Existing Azure OpenAI account in this resource group.')
param accountName string

@description('Principal to grant the role to.')
param principalId string

@description('Built-in role definition guid.')
param roleDefinitionId string

resource account 'Microsoft.CognitiveServices/accounts@2023-05-01' existing = {
  name: accountName
}

resource assignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(account.id, principalId, roleDefinitionId)
  scope: account
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roleDefinitionId)
    principalId: principalId
    principalType: 'ServicePrincipal'
  }
}

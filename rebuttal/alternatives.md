<style>
  .rebuttal-table {
    border-collapse: separate;
    border-spacing: 16px 10px; /* adjust column/row spacing (tighter between Variant & Pass@1) */
    width: 100%;
    margin-left: 0;
  }
  .rebuttal-table th,
  .rebuttal-table td {
    text-align: left;
    padding: 0;
  }
  .rebuttal-table th:nth-child(4),
  .rebuttal-table td:nth-child(4) {
    width: 24%;
  }
  .rebuttal-table th,
  .rebuttal-table td {
    vertical-align: top;
  }
  .stage-sep td {
    padding: 0;
    height: 10px;
    border-top: 2px solid #000;
  }

  .table-caption {
    text-align: center !important;
  }
  .caption-note {
    font-weight: normal;
    margin-top: 6px;
  }
</style>

<table class="rebuttal-table">
  <thead>
    <tr>
      <th colspan="4" class="table-caption">
        <div class="caption-note">Table 1. All numbers are Pass@1 on BFCL-V3 with Qwen3-8B, with each row using exactly one replacement (no combinations).</div>
      </th>
    </tr>
    <tr>
      <th>Stage</th>
      <th>Variant</th>
      <th>Pass@1</th>
      <th>Reduction vs Original</th>
    </tr>
  </thead>
  <tbody>
    <tr class="stage-sep">
      <td colspan="4"></td>
    </tr>
    <tr>
      <td>Original ExpG</td>
      <td>-</td>
      <td>67.46%</td>
      <td>0.00%</td>
    </tr>

    <tr class="stage-sep">
      <td colspan="4"></td>
    </tr>

    <tr>
      <td rowspan="2">Experience Acquisition </td>
      <td>Directly use full trajectories (no experiences)</td>
      <td>60.08% (sometimes hits context-length limits)</td>
      <td>-7.38%</td>
    </tr>
    <tr>
      <td>Directly prompt the LLM to extract experiences (no multi-aspect attribution, unstructured experiences)</td>
      <td>63.34%</td>
      <td>-4.12%</td>
    </tr>

    <tr class="stage-sep">
      <td colspan="4"></td>
    </tr>

    <tr>
      <td rowspan="4">Experience Distillation</td>
      <td>Use only the most recent 1 experience</td>
      <td>62.26%</td>
      <td>-5.20%</td>
    </tr>
    <tr>
      <td>Keep only the most recent 3 experiences</td>
      <td>64.64%</td>
      <td>-2.82%</td>
    </tr>
    <tr>
      <td>Remove the experience filter step</td>
      <td>65.94%</td>
      <td>-1.52%</td>
    </tr>
    <tr>
      <td>Remove the experience summary step</td>
      <td>63.56% (sometimes hits context-length limits)</td>
      <td>-3.90%</td>
    </tr>

    <tr class="stage-sep">
      <td colspan="4"></td>
    </tr>

    <tr>
      <td>Experience Reuse</td>
      <td>Only dynamic strategy</td>
      <td>66.37%</td>
      <td>-1.09%</td>
    </tr>
  </tbody>
</table>

